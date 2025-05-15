import argparse
import csv
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import pandas as pd
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqIO, Entrez
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


from evo2 import Evo2

def encode_sequence(sequence, tokenizer, max_length):
    return tokenizer(
        sequence,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )


def read_prompts(input_file: Path) -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []
    
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            promptseqs.append(row[0])

    return promptseqs

def translate_dna_to_protein(dna_seq: str) -> str:
    """Translate DNA sequence to protein sequence."""
    # Remove any whitespace and convert to uppercase
    dna_seq = dna_seq.strip().upper()
    
    # Create a Seq object and translate
    coding_dna = Seq(dna_seq)
    try:
        protein = str(coding_dna.translate())
        return protein
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def calculate_sequence_identity(seq1: str, seq2: str, protein: bool = False) -> Optional[float]:
    """Calculate sequence identity between two sequences using global alignment."""
    if not seq1 or not seq2:
        return None
    
    if protein:
        # Translate DNA to protein if protein comparison is requested
        seq1 = translate_dna_to_protein(seq1)
        seq2 = translate_dna_to_protein(seq2)
    
    # Define scoring parameters
    # match = 2, mismatch = -1, gap_open = -10, gap_extend = -0.5
    alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -10, -0.5)
    
    if not alignments:
        return 0.0
    
    # Get the best alignment
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, begin, end = best_alignment
    
    # Calculate identity
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
    total_length = len(aligned_seq1)  # Length including gaps
    identity = (matches / total_length) * 100
    
    return identity

def calculate_correlations(dna_similarities, protein_similarities, avg_logprobs):
    """Calculate correlations between similarities and logprobs."""
    if len(dna_similarities) < 2:
        return {
            "dna_logprobs_pearson": None,
            "dna_logprobs_spearman": None,
            "protein_logprobs_pearson": None,
            "protein_logprobs_spearman": None,
        }
    
    results = {}
    
    # Pearson correlation
    dna_logprobs_pearson, dna_p_value = pearsonr(dna_similarities, avg_logprobs)
    protein_logprobs_pearson, protein_p_value = pearsonr(protein_similarities, avg_logprobs)
    
    # Spearman correlation
    dna_logprobs_spearman, dna_spear_p = spearmanr(dna_similarities, avg_logprobs)
    protein_logprobs_spearman, protein_spear_p = spearmanr(protein_similarities, avg_logprobs)
    
    results = {
        "dna_logprobs_pearson": (dna_logprobs_pearson, dna_p_value),
        "dna_logprobs_spearman": (dna_logprobs_spearman, dna_spear_p),
        "protein_logprobs_pearson": (protein_logprobs_pearson, protein_p_value),
        "protein_logprobs_spearman": (protein_logprobs_spearman, protein_spear_p)
    }
    
    return results

def main():
    """
    Test sequence generation using the Evo2 models directly
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Generation")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_1b_base'], default='evo2_7b',
                       help="Model to test (supports evo2_7b, evo2_40b, evo2_1b_base)")
    
    args = parser.parse_args()
    
    # Set random seeds
    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)

    # Load the model and tokenizer
    model_name = "/home/ubuntu/Patho-LM/finetuned_ckpt/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 2002
    if tokenizer.model_max_length < max_length:
        tokenizer.model_max_length = max_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patho_lm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
        
    model = Evo2(args.model_name)

    tag = "|D__VIRUS;P__SSRNA;O__RETROVIRIDAE;F__LENTIVIRUS;G__HIV-1;FUNC__ENV_gp120|"
    
    # Read HIV env gene sequence from CSV
    hiv_df = pd.read_csv('./data/virus_db/HIV/patho/HIV-1.csv')
    hiv_env = hiv_df[hiv_df['gene'] == 'env']['nucleotide_sequence'].iloc[0]
    
    # Read variant env sequences from non-pathogenic viruses
    biv_df = pd.read_csv('./data/virus_db/HIV/patho/HIV-2.csv')
    fiv_df = pd.read_csv('./data/virus_db/HIV/nopatho/FIV.csv') 
    siv_df = pd.read_csv('./data/virus_db/HIV/nopatho/SIV.csv')
    
    variant_seqs = []
    for df in [biv_df, fiv_df, siv_df]:
        env_seq = df[df['gene'] == 'env']['nucleotide_sequence'].iloc[0]
        variant_seqs.append(env_seq)
        
    # Calculate sequence similarities
    for i, (virus, seq) in enumerate(zip(['HIV-2', 'FIV', 'SIV'], variant_seqs)):
        similarity = calculate_sequence_identity(hiv_env, seq)
        print(f"Similarity between HIV and {virus} env: {similarity:.2f}%")
        
    genome = hiv_env
    # Split HIV env sequence into input and target
    input_seq = genome[:len(genome)//2]  # First half of genome
    target_seq = genome[len(genome)//2:]  # Second half of genome

    # Get the two most similar variant sequences for few-shot examples
    variant_similarities = []
    for i, seq in enumerate(variant_seqs):
        similarity = calculate_sequence_identity(genome, seq)
        variant_similarities.append((i, similarity))
    
    # Sort by similarity and get top 2
    variant_similarities.sort(key=lambda x: x[1], reverse=True)
    top_2_variants = [variant_seqs[i] for i,_ in variant_similarities[:2]]
    
    # Concatenate top 2 variants for few-shot examples
    few_shot = top_2_variants[0][:] + "||" + top_2_variants[1][:] + "||"

    # Generation parameters
    num_rounds = 4  # Number of generation rounds
    bp_per_round = 128  # Generate 128bp per round
    num_beams = 8  # Number of active beams to maintain 
    seqs_per_beam = 5  # Sequences to generate per beam

    # Initial prompt
    base_prompt = f"{tag}\n{few_shot}{input_seq}"
    base_prompt = base_prompt.replace('N', '')  # Remove N characters
    current_prompt = base_prompt
    
    print("\nStarting iterative generation process with beam search:")
    print("-" * 50)
    
    # Keep track of active beams
    active_beams = [{"prompt": current_prompt, "generated": "", "cumulative_score": 0}]
    
    # Store all round data
    all_rounds_data = []
    
    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}:")
        print("-" * 30)
        
        temperature = 1.0  # Fixed temperature
        top_k = 4  # Fixed top_k
        
        # Store data for this round
        round_data = {
            "beams": [],
            "dna_similarities": [],
            "protein_similarities": [],
            "avg_logprobs": [],
            "sequences": [],
            "beam_indices": []
        }
        
        # Target slice for this round's evaluation
        current_target_slice = target_seq[round_idx*bp_per_round:(round_idx+1)*bp_per_round]
        
        # For each active beam, generate seqs_per_beam candidates
        all_candidates = []
        
        for beam_idx, beam in enumerate(active_beams):
            print(f"\nGenerating candidates for beam {beam_idx + 1}:")
            
            # Generate sequences one at a time to avoid OOM
            beam_logprobs = []
            beam_generated_seqs = []
            beam_pathogenicity_scores = []
            
            for seq_idx in range(seqs_per_beam):
                # Generate sequence one at a time
                with torch.inference_mode():
                    output = model.generate(
                        prompt_seqs=[beam["prompt"]],  # Single prompt
                        n_tokens=bp_per_round,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=0.95,
                        cached_generation=True
                    )
                
                logprob = output.logprobs_mean[0]
                generated_seq = output.sequences[0]
                
                beam_logprobs.append(logprob)
                beam_generated_seqs.append(generated_seq)
                
                # Calculate pathogenicity score
                inputs = encode_sequence(beam["generated"] + generated_seq, tokenizer, max_length)
                with torch.no_grad():
                    outputs = patho_lm(**{k: v.to(device) for k, v in inputs.items()})
                logits = outputs.logits.cpu().numpy()
                pred_class = np.argmax(logits, axis=1)[0]
                pred_label = "pathogen" if pred_class == 1 else "non-pathogen"
                print(f"Prediction: {pred_label} (class {pred_class})")
                print(f"Logits: {logits[0]}")  # Print raw logits for the sequence
                pathogenicity = logits[0][1]
                beam_pathogenicity_scores.append(pathogenicity)
                
                # Calculate DNA similarity to the current target slice
                dna_similarity = calculate_sequence_identity(generated_seq, current_target_slice)
                
                # Calculate protein similarity
                protein_similarity = calculate_sequence_identity(generated_seq, current_target_slice, protein=True)
                
                print(f"Candidate {seq_idx+1}: LogProb: {logprob:.4f}, DNA Similarity: {dna_similarity:.2f}%")
                
                # Store data
                round_data["dna_similarities"].append(dna_similarity)
                round_data["protein_similarities"].append(protein_similarity)
                round_data["avg_logprobs"].append(logprob)
                round_data["sequences"].append(generated_seq)
                round_data["beam_indices"].append(beam_idx)
            
            # After generating all sequences for this beam, calculate normalized scores
            min_logprob = min(beam_logprobs)
            max_logprob = max(beam_logprobs)
            logprob_range = max_logprob - min_logprob + 1e-8  # Avoid division by zero
            
            # Process candidates for this beam
            for i in range(seqs_per_beam):
                generated_seq = beam_generated_seqs[i]
                logprob = beam_logprobs[i]
                pathogenicity = beam_pathogenicity_scores[i]
                
                # Calculate DNA similarity again (already calculated above, but needed for scoring)
                dna_similarity = calculate_sequence_identity(generated_seq, current_target_slice)
                protein_similarity = calculate_sequence_identity(generated_seq, current_target_slice, protein=True)
                
                # Normalize scores within this beam
                normalized_similarity = dna_similarity / 100.0
                normalized_logprob = (logprob - min_logprob) / logprob_range
                
                # Store candidate
                candidate = {
                    "prompt": beam["prompt"],
                    "generated": beam["generated"] + generated_seq,
                    "last_segment": generated_seq,
                    "parent_beam": beam_idx,
                    "dna_similarity": dna_similarity,
                    "protein_similarity": protein_similarity,
                    "logprob": logprob,
                    "pathogenicity": pathogenicity,
                    "cumulative_score": beam["cumulative_score"]  # Temporary, will update after normalizing pathogenicity
                }
                
                all_candidates.append(candidate)
        
        # Now normalize pathogenicity scores across all candidates
        all_pathogenicity_scores = [c["pathogenicity"] for c in all_candidates]
        min_pathogenicity = min(all_pathogenicity_scores)
        max_pathogenicity = max(all_pathogenicity_scores)
        pathogenicity_range = max_pathogenicity - min_pathogenicity + 1e-8  # Avoid division by zero
        
        # Also normalize logprobs across all candidates
        all_logprobs = [c["logprob"] for c in all_candidates]
        min_logprob_global = min(all_logprobs)
        max_logprob_global = max(all_logprobs)
        logprob_range_global = max_logprob_global - min_logprob_global + 1e-8  # Avoid division by zero
        
        # Update all candidates with normalized scores and combined score
        for candidate in all_candidates:
            normalized_pathogenicity = (candidate["pathogenicity"] - min_pathogenicity) / pathogenicity_range
            normalized_logprob = (candidate["logprob"] - min_logprob_global) / logprob_range_global
            combined_score = normalized_logprob + normalized_pathogenicity * 0.5
            candidate["normalized_pathogenicity"] = normalized_pathogenicity
            candidate["normalized_logprob"] = normalized_logprob
            candidate["combined_score"] = combined_score
            candidate["cumulative_score"] = active_beams[candidate["parent_beam"]]["cumulative_score"] + combined_score
        
        # Add round data
        all_rounds_data.append(round_data)
        
        # Select top 'num_beams' candidates as new active beams
        all_candidates.sort(key=lambda x: x["cumulative_score"], reverse=True)
        active_beams = all_candidates[:num_beams]
        
        print("\nSelected top beams for next round:")
        for i, beam in enumerate(active_beams):
            print(f"Beam {i+1}: DNA Similarity: {beam['dna_similarity']:.2f}%, LogProb: {beam['logprob']:.4f}, Pathogenicity: {beam['pathogenicity']:.4f}")
            print(f"Normalized: LogProb: {beam['normalized_logprob']:.4f}, Pathogenicity: {beam['normalized_pathogenicity']:.4f}, Combined: {beam['combined_score']:.4f}")
            print(f"Last segment: {beam['last_segment']}")
        
        # Calculate correlations for this round
        correlations = calculate_correlations(
            round_data["dna_similarities"],
            round_data["protein_similarities"],
            round_data["avg_logprobs"]
        )
        
        print("\nCorrelations for this round:")
        print(f"DNA Similarity vs LogProbs (Pearson): r={correlations['dna_logprobs_pearson'][0]:.4f}, p={correlations['dna_logprobs_pearson'][1]:.4f}")
        print(f"DNA Similarity vs LogProbs (Spearman): r={correlations['dna_logprobs_spearman'][0]:.4f}, p={correlations['dna_logprobs_spearman'][1]:.4f}")
        print(f"Protein Similarity vs LogProbs (Pearson): r={correlations['protein_logprobs_pearson'][0]:.4f}, p={correlations['protein_logprobs_pearson'][1]:.4f}")
        print(f"Protein Similarity vs LogProbs (Spearman): r={correlations['protein_logprobs_spearman'][0]:.4f}, p={correlations['protein_logprobs_spearman'][1]:.4f}")
        
        # Update prompt for each active beam for next round
        if round_idx < num_rounds - 1:  # Don't update prompt after last round
            for beam in active_beams:
                new_input_seq = input_seq + beam["last_segment"]
                beam["prompt"] = f"{tag}\n{few_shot}{new_input_seq}"
                beam["prompt"] = beam["prompt"].replace('N', '')
            
            print(f"\nPrompt length for next round: {len(active_beams[0]['prompt'])}")
    
    # Select the best beam as the final result
    best_beam = max(active_beams, key=lambda x: x["cumulative_score"])
    full_generated_seq = best_beam["generated"]
    
    print("\nBeam search complete!")
    print(f"Total generated sequence ({len(full_generated_seq)} bp):")
    print(full_generated_seq)
    
    # Translate and print the protein sequence for input + generated
    complete_seq = input_seq + full_generated_seq
    protein_seq = translate_dna_to_protein(complete_seq)
    print("\nTranslated protein sequence:")
    print(protein_seq)
    
    # Also translate and compare with target
    target_complete = input_seq + target_seq[:len(full_generated_seq)]
    target_protein = translate_dna_to_protein(target_complete)
    print("\nTarget protein sequence:")
    print(target_protein)
    
    # Calculate protein similarity
    protein_similarity = calculate_sequence_identity(protein_seq, target_protein)
    print(f"\nProtein sequence similarity: {protein_similarity:.2f}%")
    
    # Evaluate the full generated sequence against the full target
    full_dna_similarity = calculate_sequence_identity(full_generated_seq, target_seq[:len(full_generated_seq)])
    full_protein_similarity = calculate_sequence_identity(full_generated_seq, target_seq[:len(full_generated_seq)], protein=True)
    
    print(f"\nFull sequence evaluation:")
    print(f"DNA Similarity: {full_dna_similarity:.2f}%")
    print(f"Protein Similarity: {full_protein_similarity:.2f}%")
    
    # Save all data to files
    import json
    import pickle
    
    # Save correlations across all rounds
    all_corrs = {
        "dna_pearson_r": [],
        "dna_pearson_p": [],
        "dna_spearman_r": [],
        "dna_spearman_p": [],
        "protein_pearson_r": [],
        "protein_pearson_p": [],
        "protein_spearman_r": [],
        "protein_spearman_p": []
    }
    
    for round_data in all_rounds_data:
        corrs = calculate_correlations(
            round_data["dna_similarities"],
            round_data["protein_similarities"],
            round_data["avg_logprobs"]
        )
        if corrs["dna_logprobs_pearson"] is not None:
            all_corrs["dna_pearson_r"].append(corrs["dna_logprobs_pearson"][0])
            all_corrs["dna_pearson_p"].append(corrs["dna_logprobs_pearson"][1])
            all_corrs["dna_spearman_r"].append(corrs["dna_logprobs_spearman"][0])
            all_corrs["dna_spearman_p"].append(corrs["dna_logprobs_spearman"][1])
            all_corrs["protein_pearson_r"].append(corrs["protein_logprobs_pearson"][0])
            all_corrs["protein_pearson_p"].append(corrs["protein_logprobs_pearson"][1])
            all_corrs["protein_spearman_r"].append(corrs["protein_logprobs_spearman"][0])
            all_corrs["protein_spearman_p"].append(corrs["protein_logprobs_spearman"][1])
    
    # Save correlation summary
    with open(f"evo2_generation_correlations_{args.model_name}.json", "w") as f:
        json.dump(all_corrs, f, indent=2)
    
    # Save all generation data for further analysis
    with open(f"evo2_generation_data_{args.model_name}.pkl", "wb") as f:
        pickle.dump({
            "rounds_data": all_rounds_data,
            "full_generated_sequence": full_generated_seq,
            "full_dna_similarity": full_dna_similarity,
            "full_protein_similarity": full_protein_similarity,
            "beam_paths": active_beams,
            "generated_protein": protein_seq,
            "target_protein": target_protein,
            "protein_similarity": protein_similarity
        }, f)
    
    print(f"\nSaved correlation data to evo2_generation_correlations_{args.model_name}.json")
    print(f"Saved full generation data to evo2_generation_data_{args.model_name}.pkl")

if __name__ == "__main__":
    main()