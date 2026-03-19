#!/usr/bin/env python3
"""
Amber Alert Research Classifier - "Grounded Definition" Standard
--------------------------------------------------------------
Methodology: 
  1. Grounded Definitions: Uses Table C1 "Snippets" to teach the model specific user language.
  2. Hierarchy Rule: Resolves multi-label conflicts.
  3. 20-Shot Validation: Uses real examples to stabilize output.
  4. Models: Llama 3.1, Mistral, Falcon 3.
"""

import os
import sys
import re
import gc
import torch
import pandas as pd
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# CONFIGURATION
# ---------------------------
LLAMA_ID = "meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
FALCON_ID = "tiiuae/Falcon3-7B-Instruct"  # Using the Smart Falcon 3

INPUT_CSV = "reddit_amber_alert_summary.csv"
OUT_DIR = "./outputs"
FINAL_CSV = os.path.join(OUT_DIR, "final_grounded_output.csv")
SUMMARY_FILE = os.path.join(OUT_DIR, "final_research_summary.txt")

ROW_LIMIT = None # Set to None for full run

# ---------------------------
# THE "GROUNDED" RESEARCH PROMPT
# ---------------------------
SYSTEM_INSTRUCTION = """You are a qualitative researcher coding Reddit comments based on 'Table C1: IT Capabilities'.

### HIERARCHY RULE (Tie-Breaker):
1. **High Priority:** Actionable Intel (Keynoting, Mapping, Retrieval, Threat Intel).
2. **Medium Priority:** System Mechanics (Notification Mgmt, Privacy, Updating, Multimedia).
3. **Low Priority:** Sentiment/Support (Affective Comm, Engagement, Helping).

### DEFINITIONS & SIGNAL PHRASES (From Table C1):

1. **Keynoting Capability**:
   - *Definition:* Extracting/surfacing critical details (vehicle, plate, color).
   - *Look for comments like:* "I remember the car model", "Blue sweatshirt", "License plate number".

2. **Reminder Capability**:
   - *Definition:* System sending repeats, follow-ups, or reminders to maintain engagement.
   - *Look for comments like:* "It would be nice to get an alert circling back", "Repeated alerts help", "Remind people".

3. **Multimedia Capability**:
   - *Definition:* Incorporating images, audio, and video content.
   - *Look for comments like:* "The picture helps see a face", "Add pictures", "No image included".

4. **Mapping Capability**:
   - *Definition:* Integration of geolocation, distance, search areas, or visualization.
   - *Look for comments like:* "300 miles away", "Going north through Waco", "Near my location", "Geofence".

5. **Affective Communication Capability**:
   - *Definition:* Emotional tone, motivation, emotional safety.
   - *Look for comments like:* "Makes me cry", "Feel bad for the kid", "It touches me", "So sad".

6. **Threat Intelligence Capability**:
   - *Definition:* Flagging threat levels or urgency (e.g., armed, dangerous).
   - *Look for comments like:* "Armed or dangerous", "Impacts the urgency", "I would pay more attention".

7. **Multiplatform Integration Capability**:
   - *Definition:* Syncing across TV, Social Media, Apps, Email.
   - *Look for comments like:* "Show up in social media feed", "I follow police on Facebook", "Check Twitter".

8. **Peer Sharing Capability**:
   - *Definition:* Sharing alerts with social networks/friends.
   - *Look for comments like:* "No way to share this", "Allow us to forward it", "I shared this with my group".

9. **Updating Capability**:
   - *Definition:* Real-time status changes (Found/Closed). *Not app updates.*
   - *Look for comments like:* "Tell that the child was found", "Alert to close the case", "Is there an update?".

10. **Retrieval Capability**:
    - *Definition:* Accessing old/closed messages or history.
    - *Look for comments like:* "Message is gone if you click", "Way to retrieve messages", "Look up old alerts".

11. **Rewarding Capability**:
    - *Definition:* Incentives, gamification, badges for help.
    - *Look for comments like:* "Reward system", "Prize money", "Motivates people".

12. **Tip-Verification Capability**:
    - *Definition:* Validating sightings, ensuring certainty before reporting.
    - *Look for comments like:* "Don't want to waste their time", "Need to be completely sure", "Confirm suspicions".

13. **Privacy Capability**:
    - *Definition:* **ANONYMITY** in reporting. Submitting tips without revealing identity.
    - *Look for comments like:* "Give anonymous tips", "Offer info without going to police station".
    - *NOTE:* Do NOT use this for complaints about 'intrusion'. Use Notification Management.

14. **Reporting Capability**:
    - *Definition:* Mechanisms to send info (QR Code, Feedback buttons).
    - *Look for comments like:* "Scan QR code", "Report directly", "Feedback pathway".

15. **Helping Capability**:
    - *Definition:* Guidance, FAQs, instructions on what to do.
    - *Look for comments like:* "Better understanding of what to do", "Link to get more info", "What do they expect us to do?".

16. **Notification Management Capability**:
    - *Definition:* **Sleep Disturbance**, loud noises, turning alerts ON/OFF, scheduling.
    - *Look for comments like:* "3:00 AM", "Disturbed sleep", "Just so loud", "Turned alerts off", "Freaked me out".

17. **Engagement Capability**:
    - *Definition:* Commenting, reacting, community discussion.
    - *Look for comments like:* "People would comment", "Post it and help out", "Discussion thread".

18. **Other Capability**: Technical infrastructure issues (e.g., "Link is broken", "Cell tower failure").
19. **None**: Irrelevant noise, politics, jokes, spam.

### 20 REAL-WORLD REFERENCE EXAMPLES:

1. Comment: "They know everyone has turned off Amber Alerts so they do this crap."
   Reason: User discusses disabling system functionality (matches 'Turned alerts off' snippet).
   Label: Notification Management Capability
   Confidence: 100

2. Comment: "Same !! Haha"
   Reason: Conversational noise.
   Label: None
   Confidence: 100

3. Comment: "The weather channel app can alert me of those tornadoes."
   Reason: Suggests alternative platform (matches 'Multiplatform' definition).
   Label: Multiplatform Integration Capability
   Confidence: 90

4. Comment: "You know, the guy ended up getting caught in Fort Worth."
   Reason: Provides status update on the case.
   Label: Updating Capability
   Confidence: 95

5. Comment: "If we have them all turned off then we won’t be ready when the civil war erupts."
   Reason: Political statement, unrelated to system mechanics.
   Label: None
   Confidence: 100

6. Comment: "Oh my god that's why i got it?!?! I have amber alerts turned off."
   Reason: User discusses settings/overriding 'off' switch.
   Label: Notification Management Capability
   Confidence: 100

7. Comment: "Child is in imminent danger? For 5 months?"
   Reason: Questions the urgency/threat level.
   Label: Threat Intelligence Capability
   Confidence: 85

8. Comment: "It’s because they had recently found the vehicle."
   Reason: Provides investigation details (Vehicle found).
   Label: Updating Capability
   Confidence: 90

9. Comment: "The abductor has the same last name as the child, probably domestic."
   Reason: Infers context about the people involved.
   Label: Keynoting Capability
   Confidence: 80

10. Comment: "Had all my alerts turned off except Extreme Threats... So I just forcefully uninstalled it."
    Reason: user discusses managing/removing software.
    Label: Notification Management Capability
    Confidence: 100

11. Comment: "30-40 minutes south of Dallas..."
    Reason: Specific geolocation details.
    Label: Mapping Capability
    Confidence: 95

12. Comment: "Yes, but it's not really relevant to me when a kid went missing in Houston, I'm in Lubbock."
    Reason: Complains about distance/relevance (Geofencing).
    Label: Mapping Capability
    Confidence: 90

13. Comment: "I pay my taxes now leave me alone."
    Reason: Irrelevant complaint.
    Label: None
    Confidence: 100

14. Comment: "Is there a way to do this on iPhone?"
    Reason: Asking for help/guidance on settings.
    Label: Helping Capability
    Confidence: 85

15. Comment: "The license plate was ABC-123."
    Reason: Critical detail identified.
    Label: Keynoting Capability
    Confidence: 100

16. Comment: "I hope they find the kid safe, this is so sad."
    Reason: Emotional support.
    Label: Affective Communication Capability
    Confidence: 90

17. Comment: "I looked up the old alert history."
    Reason: Accessing past data.
    Label: Retrieval Capability
    Confidence: 95

18. Comment: "Why does the link go to a broken page?"
    Reason: Technical failure.
    Label: Other Capability
    Confidence: 90

19. Comment: "I saw the car! I am calling 911."
    Reason: Taking action to report.
    Label: Reporting Capability
    Confidence: 100

20. Comment: "Why is there no picture included?"
    Reason: Requesting visual media.
    Label: Multimedia Capability
    Confidence: 95

### YOUR TASK:
Analyze the following comment.
Output strictly in this format:
Reason: <Your reasoning based on the definitions above>
Confidence: <0-100>
Label: <Exact Category Name>
"""

# ---------------------------
# UTILITIES
# ---------------------------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def normalize_label(lbl: str) -> str:
    if not lbl: return "None"
    s = re.sub(r'[^\w\s]', '', str(lbl)).strip().lower()
    valid_labels = [
        "keynoting capability", "reminder capability", "multimedia capability", 
        "mapping capability", "affective communication capability", "threat intelligence capability",
        "multiplatform integration capability", "peer sharing capability", "updating capability",
        "retrieval capability", "rewarding capability", "tip-verification capability",
        "privacy capability", "reporting capability", "helping capability",
        "notification management capability", "engagement capability", "other capability", "none"
    ]
    for v in valid_labels:
        if v in s: return v
    return "None"

def parse_strict_output(text: str):
    reason = "Parse Error"
    label = "None"
    conf = 0.0
    txt = text.strip()
    
    m_reason = re.search(r"Reason:?\s*(.*?)(?:\nLabel|Confidence|$)", txt, re.IGNORECASE | re.DOTALL)
    if m_reason: reason = m_reason.group(1).strip()
    
    m_label = re.search(r"Label:?\s*(.+)", txt, re.IGNORECASE)
    if m_label: label = m_label.group(1).splitlines()[0].strip().replace("*", "")

    m_conf = re.search(r"Confidence:?\s*([0-9]*\.?[0-9]+)", txt, re.IGNORECASE)
    if m_conf:
        try:
            val = float(m_conf.group(1))
            if val <= 1.0 and val > 0: val *= 100.0
            conf = min(100.0, max(0.0, val))
        except: conf = 0.0
    return reason, label, conf

def load_model_tokenizer(identifier):
    print(f">>> Loading {identifier}...")
    try:
        # Ensure trust_remote_code is False for stability
        tok = AutoTokenizer.from_pretrained(identifier, use_fast=False, trust_remote_code=False)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            identifier, device_map="auto", torch_dtype=torch.float16, trust_remote_code=False
        )
        return model, tok
    except Exception as e:
        print(f"Error: {e}"); return None, None

def safe_unload(model, tokenizer):
    try: del model; del tokenizer
    except: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ---------------------------
# INFERENCE
# ---------------------------
def run_model_inference(model_name, model, tokenizer, comments):
    out = []
    device = next(model.parameters()).device
    has_chat = hasattr(tokenizer, "apply_chat_template")
    
    for comment in tqdm(comments, desc=f"{model_name}"):
        clean = comment.replace("\n", " ").strip()
        
        # Prompt construction with definitions and examples
        user_msg = "Classify based on the definitions and examples above."
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, 
                    {"role": "user", "content": f"Comment: \"{clean}\"\n{user_msg}"}]
        
        if has_chat: 
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{SYSTEM_INSTRUCTION}\n\nComment: \"{clean}\"\n{user_msg}\nOutput:"
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
        
        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                reason, label, conf = parse_strict_output(decoded)
                out.append({"label": label, "reason": reason, "conf": conf})
            except Exception as e:
                out.append({"label": "None", "reason": f"Error: {e}", "conf": 0.0})
    return out

# ---------------------------
# REPORTING
# ---------------------------
def generate_report(results):
    df = pd.DataFrame(results)
    total = len(df)
    majority = df[df["Agreement_Score"] >= 66.0]
    consensus = (len(majority) / total) * 100
    perfect = df[df["Agreement_Score"] == 100.0]
    perf_rate = (len(perfect) / total) * 100
    top = df['Final_Consensus_Label'].value_counts().head(5)

    report = []
    report.append("="*60)
    report.append("RESEARCH REPORT: GROUNDED ENSEMBLE")
    report.append("="*60)
    report.append(f"Total Comments:      {total}")
    report.append(f"Majority Consensus:  {consensus:.2f}%")
    report.append(f"Perfect Agreement:   {perf_rate:.2f}%")
    report.append("-" * 60)
    report.append("TOP 5 CAPABILITIES:")
    for l, c in top.items(): report.append(f"  - {l}: {c}")
    report.append("="*60)
    
    print("\n".join(report))
    with open(SUMMARY_FILE, "w") as f: f.write("\n".join(report))

# ---------------------------
# MAIN
# ---------------------------
def main():
    ensure_outdir()
    if not os.path.exists(INPUT_CSV): print(f"Error: {INPUT_CSV}"); sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    if ROW_LIMIT: df = df.head(ROW_LIMIT)
    comments = [str(c) for c in df['comment'].fillna("").tolist()]
    
    # 1. Llama 3.1
    lm, lt = load_model_tokenizer(LLAMA_ID)
    l_res = run_model_inference("Llama", lm, lt, comments) if lm else [{"label":"None","conf":0,"reason":"Fail"}]*len(comments)
    safe_unload(lm, lt)
    
    # 2. Mistral v0.3
    mm, mt = load_model_tokenizer(MISTRAL_PATH)
    m_res = run_model_inference("Mistral", mm, mt, comments) if mm else [{"label":"None","conf":0,"reason":"Fail"}]*len(comments)
    safe_unload(mm, mt)
    
    # 3. Falcon 3 (Smart)
    fm, ft = load_model_tokenizer(FALCON_ID)
    f_res = run_model_inference("Falcon", fm, ft, comments) if fm else [{"label":"None","conf":0,"reason":"Fail"}]*len(comments)
    safe_unload(fm, ft)

    # 4. Voting
    print("\n>>> Calculating Consensus...")
    final_results = []
    for i in range(len(comments)):
        l_lbl = normalize_label(l_res[i]["label"])
        m_lbl = normalize_label(m_res[i]["label"])
        f_lbl = normalize_label(f_res[i]["label"])
        
        votes = [l_lbl, m_lbl, f_lbl]
        counts = Counter(votes)
        winner, win_count = counts.most_common(1)[0]
        
        final_label = winner if win_count >= 2 else l_lbl 
        agree = round((win_count/3)*100, 2)
        
        final_results.append({
            "comment": comments[i],
            "Llama_Label": l_lbl, "Llama_Reason": l_res[i]["reason"],
            "Mistral_Label": m_lbl, "Mistral_Reason": m_res[i]["reason"],
            "Falcon_Label": f_lbl, "Falcon_Reason": f_res[i]["reason"],
            "Final_Consensus_Label": final_label,
            "Agreement_Score": agree
        })

    pd.DataFrame(final_results).to_csv(FINAL_CSV, index=False)
    generate_report(final_results)
    print(f"Done. Results: {FINAL_CSV}")

if __name__ == "__main__":
    main()
