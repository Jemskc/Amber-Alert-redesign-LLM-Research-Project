#!/usr/bin/env python3
"""
Amber Alert A100 Final - Chunk 1 (User Profile + Dynamic 'Other')
-----------------------------------------------------------------
1. Feature: Added 'User Profile & Personalization Capability'.
2. Feature: Allows 'Other Capability (Specific Issue)' labels.
3. Hardware: A100 Optimized (Bfloat16 + SDPA + Batch 32).
4. Input: ./split_datasets/chunk_1.csv
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

# --- CRITICAL MEMORY SETTING ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------
# CONFIGURATION
# ---------------------------
LLAMA_ID = "meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
FALCON_ID = "tiiuae/Falcon3-7B-Instruct"

# FILES
INPUT_CSV = "./split_datasets/chunk_1.csv"
OUT_DIR = "./outputs_new_gpw"

# OUTPUTS
FINAL_CSV = os.path.join(OUT_DIR, "final_gpu_1_analysis.csv")
LLAMA_TEMP = os.path.join(OUT_DIR, "temp_llama_checkpoint.csv")
MISTRAL_TEMP = os.path.join(OUT_DIR, "temp_mistral_checkpoint.csv")
FALCON_TEMP = os.path.join(OUT_DIR, "temp_falcon_checkpoint.csv")

# SPEED SETTINGS
BATCH_SIZE = 16
ROW_LIMIT = None

# ---------------------------
# UPDATED RESEARCH PROMPT
# ---------------------------
SYSTEM_INSTRUCTION = """You are a qualitative researcher coding Reddit comments based on 'Table C1: IT Capabilities'.

### CORE TASK:
Analyze the comment. Determine if it discusses ONE capability or MULTIPLE capabilities.
- **Natural Count:** List strictly what is present. 
- If multiple apply, separate them using "||".

### DEFINITIONS & SIGNAL PHRASES (Table C1):

1. **Keynoting Capability**: 
   - *Definition:* Extracting/surfacing critical details (vehicle, plate, color). 
   - *Signals:* "I remember the car model", "Blue sweatshirt", "License plate number".

2. **Reminder Capability**: 
   - *Definition:* System sending repeats, follow-ups, or reminders to maintain engagement.
   - *Signals:* "It would be nice to get an alert circling back", "Repeated alerts help", "Remind people".

3. **Multimedia Capability**: 
   - *Definition:* Incorporating images, audio, and video content.
   - *Signals:* "The picture helps see a face", "Add pictures", "No image included".

4. **Mapping Capability**: 
   - *Definition:* Integration of geolocation, distance, search areas, or visualization.
   - *Signals:* "300 miles away", "Going north through Waco", "Near my location", "Geofence".

5. **Affective Communication Capability**: 
   - *Definition:* Emotional tone, motivation, emotional safety.
   - *Signals:* "Makes me cry", "Feel bad for the kid", "It touches me", "So sad".

6. **Threat Intelligence Capability**: 
   - *Definition:* Flagging threat levels or urgency (e.g., armed, dangerous).
   - *Signals:* "Armed or dangerous", "Impacts the urgency", "I would pay more attention".

7. **Multiplatform Integration Capability**: 
   - *Definition:* Syncing across TV, Social Media, Apps, Email.
   - *Signals:* "Show up in social media feed", "I follow police on Facebook", "Check Twitter".

8. **Peer Sharing Capability**: 
   - *Definition:* Sharing alerts with social networks/friends.
   - *Signals:* "No way to share this", "Allow us to forward it", "I shared this with my group".

9. **Updating Capability**: 
   - *Definition:* Real-time status changes (Found/Closed). *Not app updates.*
   - *Signals:* "Tell that the child was found", "Alert to close the case", "Is there an update?".

10. **Retrieval Capability**: 
    - *Definition:* Accessing old/closed messages or history.
    - *Signals:* "Message is gone if you click", "Way to retrieve messages", "Look up old alerts".

11. **Rewarding Capability**: 
    - *Definition:* Incentives, gamification, badges for help.
    - *Signals:* "Reward system", "Prize money", "Motivates people".

12. **Tip-Verification Capability**: 
    - *Definition:* Validating sightings, ensuring certainty before reporting.
    - *Signals:* "Don't want to waste their time", "Need to be completely sure", "Confirm suspicions".

13. **Privacy Capability**: 
    - *Definition:* **ANONYMITY** in reporting. Submitting tips without revealing identity.
    - *Signals:* "Give anonymous tips", "Offer info without going to police station".
    - *NOTE:* Do NOT use this for complaints about 'intrusion'. Use Notification Management.

14. **Reporting Capability**: 
    - *Definition:* Mechanisms to send info (QR Code, Feedback buttons).
    - *Signals:* "Scan QR code", "Report directly", "Feedback pathway".

15. **Helping Capability**: 
    - *Definition:* Guidance, FAQs, instructions on what to do.
    - *Signals:* "Better understanding of what to do", "Link to get more info", "What do they expect us to do?".

16. **Notification Management Capability**: 
    - *Definition:* **Sleep Disturbance**, loud noises, turning alerts ON/OFF, scheduling.
    - *Signals:* "3:00 AM", "Disturbed sleep", "Just so loud", "Turned alerts off", "Freaked me out".

17. **User Profile & Personalization Capability**:
    - *Definition:* Customizing settings, language preferences, radius, or specific alert types.
    - *Signals:* "Customize my radius", "Set language to Spanish", "Only show me local alerts", "Personal settings".

18. **Engagement Capability**: 
    - *Definition:* Commenting, reacting, community discussion.
    - *Signals:* "People would comment", "Post it and help out", "Discussion thread".

19. **Other Capability**: 
    - *Definition:* Technical infrastructure issues not covered above.
    - *INSTRUCTION:* If selected, you MUST include the specific issue in parentheses. 
    - *Example:* "Other Capability (Broken Link)", "Other Capability (Battery Drain)".

20. **None**: Irrelevant noise, politics, jokes, spam.

### 20 REFERENCE EXAMPLES (LOGIC GUIDE):

1. Comment: "They know everyone has turned off Amber Alerts so they do this crap."
   Reason: User discusses disabling system functionality.
   Label: Notification Management Capability
   Confidence: 100

2. Comment: "Same !! Haha"
   Reason: Conversational noise.
   Label: None
   Confidence: 100

3. Comment: "The weather channel app can alert me of those tornadoes."
   Reason: Suggests alternative platform.
   Label: Multiplatform Integration Capability
   Confidence: 90

4. Comment: "You know, the guy ended up getting caught in Fort Worth."
   Reason: Provides status update on the case.
   Label: Updating Capability
   Confidence: 95

5. Comment: "If we have them all turned off then we won’t be ready when the civil war erupts."
   Reason: Political statement.
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
    Label: Other Capability (Broken Link)
    Confidence: 90

19. Comment: "I saw the car! I am calling 911."
    Reason: Taking action to report.
    Label: Reporting Capability
    Confidence: 100

20. Comment: "Why is there no picture included?"
    Reason: Requesting visual media.
    Label: Multimedia Capability
    Confidence: 95

### OUTPUT FORMATTING RULES (HOW TO USE ||):

**Example A (Multi-Label):**
Comment: "I saw the grey Honda on I-35 but I hate how loud the alert was."
Output:
Label: Keynoting Capability
Confidence: 100
Reason: User identifies specific vehicle details (Grey Honda).
||
Label: Notification Management Capability
Confidence: 95
Reason: User complains about the loud noise/volume.

**Example B (Multi-Label):**
Comment: "I clicked the link to see the map but it was broken."
Output:
Label: Mapping Capability
Confidence: 90
Reason: User attempting to access map/location.
||
Label: Other Capability (Technical Glitch)
Confidence: 95
Reason: Reporting a technical failure.

### YOUR TASK:
Analyze the following comment. 
List strictly what is present based on the definitions and 20 examples above.
Use "||" to separate blocks if multiple apply.
**IMPORTANT:** Do not simply copy reasons from the examples. Write a unique reason based on the specific text of the comment provided.

Comment: "{comment}"

OUTPUT FORMAT:
Label: <Exact Category Name>
Confidence: <0-100>
Reason: <Your unique analysis>
||
Label: <Next Category Name>
...
"""

# ---------------------------
# UTILITIES
# ---------------------------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def normalize_label(lbl: str) -> str:
    """
    Standardizes labels. 
    SPECIAL LOGIC: If label is 'Other Capability (detail)', it is kept.
    """
    if not lbl: return "None"
    s = str(lbl).strip()
    s_lower = s.lower()
    
    # 1. ALLOW 'Other Capability (...)' PASSTHROUGH
    if s_lower.startswith("other capability"):
        return s  # Return full string including parentheses

    # 2. STANDARD CHECK FOR OTHERS
    # Remove punctuation for strict matching on standard labels
    clean_s = re.sub(r'[^\w\s]', '', s).strip().lower()
    
    valid_labels = [
        "keynoting capability", "reminder capability", "multimedia capability", 
        "mapping capability", "affective communication capability", "threat intelligence capability",
        "multiplatform integration capability", "peer sharing capability", "updating capability",
        "retrieval capability", "rewarding capability", "tip-verification capability",
        "privacy capability", "reporting capability", "helping capability",
        "notification management capability", "engagement capability", 
        "user profile & personalization capability", # ADDED
        "none"
    ]
    
    for v in valid_labels:
        # Match exact clean string
        if v == clean_s: return v
        # Fuzzy match
        if v in clean_s and len(clean_s) > len(v) * 0.8: return v
        
    return "None"

def parse_multilabel_output(text: str):
    results = []
    blocks = text.split("||")
    for block in blocks:
        clean = block.strip()
        if not clean: continue
        
        m_lbl = re.search(r"Label:?\s*(.+)", clean, re.IGNORECASE)
        m_conf = re.search(r"Confidence:?\s*([0-9]*\.?[0-9]+)", clean, re.IGNORECASE)
        m_reas = re.search(r"Reason:?\s*(.*?)(?:\nLabel|Confidence|$)", clean, re.IGNORECASE | re.DOTALL)
        
        if m_lbl:
            raw_label = m_lbl.group(1).strip()
            # Normalize handles the 'Other Capability (detail)' logic
            label = normalize_label(raw_label)
            
            conf = 0.0
            if m_conf:
                try:
                    val = float(m_conf.group(1))
                    conf = min(100.0, max(0.0, val * 100.0 if val <= 1.0 else val))
                except: pass
            
            reason = m_reas.group(1).strip() if m_reas else "Parsed"
            
            # Avoid duplicates within one model response
            if not any(d['label'].lower() == label.lower() for d in results):
                results.append({"label": label, "conf": conf, "reason": reason})
                
    if not results: results.append({"label": "None", "conf": 0.0, "reason": "None"})
    return results

def format_csv(res):
    if not isinstance(res, list): return "None", "0", "None"
    res.sort(key=lambda x: (x['label'] == 'None', x['label']))
    l = " || ".join([str(x['label']) for x in res])
    c = " || ".join([str(x['conf']) for x in res])
    r = " || ".join([str(x['reason']) for x in res])
    return l, c, r

def load_model_tokenizer(identifier):
    print(f">>> Loading {identifier} in Bfloat16 (A100 Native)...")
    aggressive_gc()
    try:
        tok = AutoTokenizer.from_pretrained(identifier, use_fast=False, trust_remote_code=False, padding_side="left")
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        
        # A100 Optimized Loading
        model = AutoModelForCausalLM.from_pretrained(
            identifier, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa", 
            trust_remote_code=False
        )
        return model, tok
    except Exception as e:
        print(f"Error loading {identifier}: {e}")
        aggressive_gc()
        return None, None

def safe_unload(model, tokenizer):
    try: del model; del tokenizer
    except: pass
    aggressive_gc()

def aggressive_gc():
    gc.collect()
    torch.cuda.empty_cache()
    try: torch.cuda.ipc_collect()
    except: pass

# ---------------------------
# BATCH INFERENCE
# ---------------------------
def process_batch_safe(model, tokenizer, batch_prompts, device):
    """Attempts batch processing. Falls back to sequential on OOM."""
    try:
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            input_len = inputs['input_ids'].shape[1]
            generated = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            return [parse_multilabel_output(txt) for txt in decoded]
    except torch.cuda.OutOfMemoryError:
        print("\n[WARN] OOM Detected! Switching to sequential mode...")
        aggressive_gc()
        results = []
        for prompt in batch_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    dec = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    results.append(parse_multilabel_output(dec))
            except Exception as e2:
                results.append([{"label": "None", "conf": 0, "reason": "Error"}])
        return results

def run_inference_loop(model_name, model, tokenizer, comments, checkpoint_file):
    processed_results = []
    if os.path.exists(checkpoint_file):
        print(f"[{model_name}] Resuming from checkpoint: {checkpoint_file}")
        try:
            df_exist = pd.read_csv(checkpoint_file)
            for _, row in df_exist.iterrows():
                lbls = str(row['label']).split(" || ")
                confs = str(row['conf']).split(" || ")
                reas = str(row['reason']).split(" || ")
                items = []
                for j in range(len(lbls)):
                    items.append({"label": lbls[j], "conf": confs[j] if j<len(confs) else 0, "reason": reas[j] if j<len(reas) else ""})
                processed_results.append(items)
            print(f"   -> Already processed: {len(processed_results)} rows.")
        except:
            print("   -> Checkpoint read error. Starting fresh.")
            processed_results = []
    
    start_idx = len(processed_results)
    if start_idx >= len(comments):
        print(f"[{model_name}] Completed previously.")
        return processed_results

    if model is None:
        print(f"[ERROR] {model_name} not loaded.")
        return processed_results

    remaining = comments[start_idx:]
    device = next(model.parameters()).device
    has_chat = hasattr(tokenizer, "apply_chat_template")
    
    prompts = []
    for c in remaining:
        clean = str(c).replace("\n", " ").strip()
        msgs = [{"role": "system", "content": SYSTEM_INSTRUCTION.format(comment=clean)},
                {"role": "user", "content": "Analyze and list ALL capabilities using ||."}]
        if has_chat: txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else: txt = f"{SYSTEM_INSTRUCTION.format(comment=clean)}\n\nUser: Output:\n"
        prompts.append(txt)

    print(f"[{model_name}] Processing {len(prompts)} items...")
    
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{model_name}"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_outputs = process_batch_safe(model, tokenizer, batch_prompts, device)
        
        chunk_data = []
        for res_list in batch_outputs:
            l, c, r = format_csv(res_list)
            chunk_data.append({"label": l, "conf": c, "reason": r})
        
        pd.DataFrame(chunk_data).to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)
        
        # Keep in memory
        processed_results.extend(batch_outputs)

    return processed_results

# ---------------------------
# CONSENSUS LOGIC
# ---------------------------
def calculate_consensus(l_list, m_list, f_list):
    all_found_labels = set()
    for res in [l_list, m_list, f_list]:
        if isinstance(res, list):
            for item in res: all_found_labels.add(item['label'])
    final = []
    for label in all_found_labels:
        votes = 0
        reasons, confs = [], []
        for name, lst in [("[Llama]", l_list), ("[Mistral]", m_list), ("[Falcon]", f_list)]:
            if not isinstance(lst, list): continue
            
            # Custom match for "Other Capability (Details)"
            # We match the prefix "other capability" to count votes, 
            # but keep the specific detail in the final output.
            match = None
            if label.lower().startswith("other capability"):
                match = next((x for x in lst if x['label'].lower().startswith("other capability")), None)
            else:
                match = next((x for x in lst if x['label'] == label), None)
            
            if match:
                votes += 1
                reasons.append(f"{name}: {match['reason']}")
                confs.append(str(match['conf']))
        
        if votes >= 2:
            final.append({"label": label, "conf": " || ".join(confs), "reason": " || ".join(reasons)})
            
    has_real = any(x['label'].lower() != "none" for x in final)
    if has_real: final = [x for x in final if x['label'].lower() != "none"]
    if not final: final.append({"label": "None", "conf": "0.0", "reason": "No Consensus"})
    return final

# ---------------------------
# MAIN FLOW
# ---------------------------
def main():
    ensure_outdir()
    if not os.path.exists(INPUT_CSV): print(f"Error: {INPUT_CSV}"); sys.exit(1)

    print(">>> Loading Original Data...")
    df_main = pd.read_csv(INPUT_CSV)
    if ROW_LIMIT: df_main = df_main.head(ROW_LIMIT)
    
    comments = [str(c) for c in df_main['comment'].fillna("").tolist()]
    print(f"Total Rows: {len(comments)}")

    # 1. LLAMA
    l_res = []
    need_load = True
    if os.path.exists(LLAMA_TEMP):
        try:
            if len(pd.read_csv(LLAMA_TEMP)) >= len(comments): need_load = False
        except: pass
    
    if need_load:
        lm, lt = load_model_tokenizer(LLAMA_ID)
        l_res = run_inference_loop("Llama", lm, lt, comments, LLAMA_TEMP)
        safe_unload(lm, lt)
    else:
        print("[Llama] Checkpoint complete. Loading file...")
        df_l = pd.read_csv(LLAMA_TEMP)
        for _, row in df_l.iterrows():
            lbls = str(row['label']).split(" || ")
            confs = str(row['conf']).split(" || ")
            reas = str(row['reason']).split(" || ")
            items = []
            for j in range(len(lbls)):
                items.append({"label": lbls[j], "conf": confs[j] if j<len(confs) else 0, "reason": reas[j] if j<len(reas) else ""})
            l_res.append(items)

    # 2. MISTRAL
    m_res = []
    need_load = True
    if os.path.exists(MISTRAL_TEMP):
        try:
            if len(pd.read_csv(MISTRAL_TEMP)) >= len(comments): need_load = False
        except: pass
        
    if need_load:
        mm, mt = load_model_tokenizer(MISTRAL_PATH)
        m_res = run_inference_loop("Mistral", mm, mt, comments, MISTRAL_TEMP)
        safe_unload(mm, mt)
    else:
        print("[Mistral] Checkpoint complete. Loading file...")
        df_m = pd.read_csv(MISTRAL_TEMP)
        for _, row in df_m.iterrows():
            lbls = str(row['label']).split(" || ")
            confs = str(row['conf']).split(" || ")
            reas = str(row['reason']).split(" || ")
            items = []
            for j in range(len(lbls)):
                items.append({"label": lbls[j], "conf": confs[j] if j<len(confs) else 0, "reason": reas[j] if j<len(reas) else ""})
            m_res.append(items)

    # 3. FALCON
    f_res = []
    need_load = True
    if os.path.exists(FALCON_TEMP):
        try:
            if len(pd.read_csv(FALCON_TEMP)) >= len(comments): need_load = False
        except: pass

    if need_load:
        fm, ft = load_model_tokenizer(FALCON_ID)
        f_res = run_inference_loop("Falcon", fm, ft, comments, FALCON_TEMP)
        safe_unload(fm, ft)
    else:
        print("[Falcon] Checkpoint complete. Loading file...")
        df_f = pd.read_csv(FALCON_TEMP)
        for _, row in df_f.iterrows():
            lbls = str(row['label']).split(" || ")
            confs = str(row['conf']).split(" || ")
            reas = str(row['reason']).split(" || ")
            items = []
            for j in range(len(lbls)):
                items.append({"label": lbls[j], "conf": confs[j] if j<len(confs) else 0, "reason": reas[j] if j<len(reas) else ""})
            f_res.append(items)

    # 4. MERGE
    print("\n>>> Merging Results...")
    min_len = min(len(l_res), len(m_res), len(f_res), len(df_main))
    df_final = df_main.iloc[:min_len].copy()
    
    l_l, l_c, l_r, m_l, m_c, m_r, f_l, f_c, f_r = [],[],[],[],[],[],[],[],[]
    c_l, c_c, c_r, scores = [],[],[],[]

    for i in tqdm(range(min_len), desc="Finalizing"):
        ll, lc, lr = format_csv(l_res[i])
        ml, mc, mr = format_csv(m_res[i])
        fl, fc, fr = format_csv(f_res[i])
        cons = calculate_consensus(l_res[i], m_res[i], f_res[i])
        cl, cc, cr = format_csv(cons)
        
        score = 100.0 if (ll == ml == fl) else (66.67 if cl != "None" else 33.33)

        l_l.append(ll); l_c.append(lc); l_r.append(lr)
        m_l.append(ml); m_c.append(mc); m_r.append(mr)
        f_l.append(fl); f_c.append(fc); f_r.append(fr)
        c_l.append(cl); c_c.append(cc); c_r.append(cr)
        scores.append(score)

    df_final["Final_Consensus_Label"] = c_l
    df_final["Final_Consensus_Confidence"] = c_c
    df_final["Final_Consensus_Reason"] = c_r
    df_final["Agreement_Score"] = scores
    
    df_final["Llama_Label"] = l_l; df_final["Llama_Confidence"] = l_c; df_final["Llama_Reason"] = l_r
    df_final["Mistral_Label"] = m_l; df_final["Mistral_Confidence"] = m_c; df_final["Mistral_Reason"] = m_r
    df_final["Falcon_Label"] = f_l; df_final["Falcon_Confidence"] = f_c; df_final["Falcon_Reason"] = f_r

    df_final.to_csv(FINAL_CSV, index=False)
    print(f"DONE. Saved to: {FINAL_CSV}")

    all_lbl = []
    for r in c_l: all_lbl.extend(r.split(" || "))
    print("\nTOP 5 CAPABILITIES:")
    for k, v in Counter(all_lbl).most_common(5): print(f"{k}: {v}")

if __name__ == "__main__":
    main()
