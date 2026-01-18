import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import json
import torch
from typing import List, Any, Dict, Optional
from dataclasses import asdict
import os

# Import your existing classes
from decode_algorithm import OneStepDebugInfo, DecodeConfig
from transformers import AutoTokenizer

def linechart_confidence_trend(debug_info: List[OneStepDebugInfo], output_file: str, **kwargs) -> None:
    """
    Draw a line chart showing the relationship between decoding steps and confidence.
    
    Args:
        debug_info: List of debug information from decoding steps
        output_file: Path to save the output image
        **kwargs: Additional parameters including:
            show_stages: If True, mark stages where pass_cache=False
            figsize: Tuple for figure size
            dpi: Image resolution
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)), dpi=kwargs.get('dpi', 100))
    
    # Extract confidence data
    steps = list(range(len(debug_info)))
    mean_confidences = []
    total_confidences = []
    # min_confidences = []
    # max_confidences = []
    
    for step_info in debug_info:
        if step_info.decode_confidence and len(step_info.decode_confidence) > 0:
            # Flatten all confidence values for this step
            all_confidences = torch.cat([conf.flatten() for conf in step_info.decode_confidence if conf is not None])
            if len(all_confidences) > 0:
                mean_confidences.append(all_confidences.float().mean().item())
                total_confidences.extend(all_confidences.float().tolist())
                # min_confidences.append(all_confidences.float().min().item())
                # max_confidences.append(all_confidences.float().max().item())
            else:
                mean_confidences.append(0)
                # min_confidences.append(0)
                # max_confidences.append(0)
        else:
            mean_confidences.append(0)
            # min_confidences.append(0)
            # max_confidences.append(0)
    
    # Plot confidence trends
    ax.plot(steps, mean_confidences, 'b-', linewidth=2, label='Mean Confidence', marker='o', markersize=4)
    ax.legend()
    # ax.fill_between(steps, min_confidences, max_confidences, alpha=0.3, label='Confidence Range')
    
    # Mark stages if required
    if kwargs.get('show_stages', False):
        stage_starts = []
        for i, step_info in enumerate(debug_info):
            if hasattr(step_info, 'pass_cache') and not step_info.pass_cache:
                stage_starts.append(i)
        
        # Add vertical lines and background colors for stages
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for i, stage_start in enumerate(stage_starts):
            stage_end = stage_starts[i + 1] if i + 1 < len(stage_starts) else len(debug_info)
            color = colors[i % len(colors)]
            ax.axvspan(stage_start, stage_end, alpha=0.2, color=color, label=f'Stage {i+1}')
            
            # Add vertical line
            ax.axvline(x=stage_start, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Decoding Steps', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Relations of Confidence and Decoding Steps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
    plt.close()
    print(f"Confidence trend chart saved to: {output_file}")
    print(f"Mean confidence across {len(mean_confidences)} steps is {sum(total_confidences) / len(total_confidences):.4f}")


def boxplot_confidence_distribution(debug_info: List[List[OneStepDebugInfo]], output_file: str, **kwargs) -> None:
    """
    Draw box plots showing confidence distribution across decoding steps for multiple runs.
    
    Args:
        debug_info: List of debug information lists from multiple decoding runs
        output_file: Path to save the output image
        **kwargs: Additional parameters including:
            figsize: Tuple for figure size
            dpi: Image resolution
            max_steps: Maximum number of steps to display
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (14, 8)), dpi=kwargs.get('dpi', 100))
    
    # Collect confidence data for each step across all runs
    max_steps = max(len(x) for x in debug_info)
    step_confidences = [[] for _ in range(max_steps+1)]
    
    for run_info in debug_info:
        for step, step_info in enumerate(run_info):
            if step_info.decode_confidence and len(step_info.decode_confidence) > 0:
                # Flatten all confidence values for this step
                all_confidences = [conf.flatten() for conf in step_info.decode_confidence if conf is not None]
                if len(all_confidences) > 0:
                    step_confidences[step].extend(all_confidences)
    
    # Filter out empty steps
    valid_steps = [i for i, confs in enumerate(step_confidences) if len(confs) > 0]
    valid_confidences = [torch.cat(step_confidences[i]).float().cpu().numpy() for i in valid_steps]
    
    # Create box plot
    box_plot = ax.boxplot(valid_confidences, positions=valid_steps, patch_artist=True, 
                         showfliers=kwargs.get('show_outliers', True))
    
    # Customize box plot colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_steps)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Decoding Steps', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Confidence Distribution Across Decoding Steps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add color bar to show step progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(valid_steps)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Step Progression', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution box plot saved to: {output_file}")


def print_decode_process(debug_info: List[OneStepDebugInfo], tokenizer: Optional[AutoTokenizer]) -> None:
    """
    Print the decoding process step by step showing token evolution.
    
    Args:
        debug_info: List of debug information from decoding steps
        tokenizer: Tokenizer to handle output token id
    """
    for step, step_info in enumerate(debug_info):
        output = step_info.token_after_decode
        if tokenizer is not None:
            output = tokenizer.batch_decode(output)
        print(f'STEP {step}: {output}')

def print_window_trend(debug_info: List[OneStepDebugInfo]) -> None:
    """
    Print the window step by step showing its change.
    
    Args:
        debug_info: List of debug information from decoding steps
    """
    for step, step_info in enumerate(debug_info):
        print(f'STEP {step}: {step_info.window_range}')

def analyze_debug_infos(debug_info: List[List[OneStepDebugInfo]], output_file: str, **kwargs) -> None:
    """
    Analyze and summarize statistics from multiple debug information runs.
    
    Args:
        debug_info: List of debug information lists from multiple decoding runs
        output_file: Path to save the JSON output
        **kwargs: Additional parameters including num_confidence_intervals for confidence distribution
    """
    # Initialize distributions
    decoding_steps_dist = {}
    cache_refresh_dist = {}
    forward_length_dist = {}
    window_length_dist = {}
    decoding_tokens_per_step_dist = {}
    
    # Initialize confidence collection
    all_confidence_values = []
    
    # Collect statistics
    total_decoding_steps = 0
    total_cache_refresh = 0
    total_forward_length = 0
    total_window_length = 0
    total_decoding_tokens = 0
    total_confidence_sum = 0.0
    total_confidence_count = 0
    total_runs = len(debug_info)
    
    for run_info in debug_info:
        # Decoding steps per run
        steps = len(run_info)
        total_decoding_steps += steps
        decoding_steps_dist[steps] = decoding_steps_dist.get(steps, 0) + 1
        
        # Cache refresh count
        cache_refresh_count = sum(1 for step_info in run_info if not step_info.pass_cache)
        total_cache_refresh += cache_refresh_count
        cache_refresh_dist[cache_refresh_count] = cache_refresh_dist.get(cache_refresh_count, 0) + 1
        
        # Process each step in the run
        for step_info in run_info:
            # Forward length
            fl = step_info.forward_length
            total_forward_length += fl
            forward_length_dist[fl] = forward_length_dist.get(fl, 0) + 1
            
            # Window length
            window_len = step_info.window_range[1] - step_info.window_range[0]
            total_window_length += window_len
            window_length_dist[window_len] = window_length_dist.get(window_len, 0) + 1
            
            # Decoding tokens per step
            if step_info.decode_indices and len(step_info.decode_indices) > 0:
                token_count = sum(indices.numel() for indices in step_info.decode_indices if indices is not None)
                total_decoding_tokens += token_count
                decoding_tokens_per_step_dist[token_count] = decoding_tokens_per_step_dist.get(token_count, 0) + 1
                
                # Collect confidence values
                if step_info.decode_confidence and len(step_info.decode_confidence) > 0:
                    for conf_tensor in step_info.decode_confidence:
                        if conf_tensor is not None:
                            # Flatten the tensor and convert to list
                            conf_values = conf_tensor.flatten().tolist()
                            all_confidence_values.extend(conf_values)
                            total_confidence_sum += sum(conf_values)
                            total_confidence_count += len(conf_values)
    
    # Calculate averages
    total_steps_all_runs = sum(len(run) for run in debug_info)
    avg_decoding_steps = total_decoding_steps / total_runs if total_runs > 0 else 0
    avg_cache_refresh = total_cache_refresh / total_runs if total_runs > 0 else 0
    avg_forward_length = total_forward_length / total_steps_all_runs if total_steps_all_runs > 0 else 0
    avg_window_length = total_window_length / total_steps_all_runs if total_steps_all_runs > 0 else 0
    avg_decoding_tokens_per_step = total_decoding_tokens / total_steps_all_runs if total_steps_all_runs > 0 else 0
    avg_confidence = total_confidence_sum / total_confidence_count if total_confidence_count > 0 else 0
    
    # Calculate confidence distribution
    num_intervals = kwargs.get("num_confidence_intervals", 10)
    confidence_dist = {}
    
    if all_confidence_values:
        interval_size = 1.0 / num_intervals
        
        for i in range(num_intervals):
            lower_bound = i * interval_size
            upper_bound = (i + 1) * interval_size
            
            # Handle the last interval to include 1.0
            if i == num_intervals - 1:
                count = sum(1 for conf in all_confidence_values if lower_bound <= conf <= upper_bound)
            else:
                count = sum(1 for conf in all_confidence_values if lower_bound <= conf < upper_bound)
            
            key = f"{lower_bound:.3f}~{upper_bound:.3f}"
            confidence_dist[key] = count
    
    def sort_dict(dct):
        return {x:y for x,y in sorted(dct.items(), key=lambda d: d[0])}

    # Build analysis results
    analysis_results = {
        "avg_decoding_steps": avg_decoding_steps,
        "avg_cache_refresh": avg_cache_refresh,
        "avg_forward_length": avg_forward_length,
        "avg_window_length": avg_window_length,
        "avg_decoding_tokens_per_step": avg_decoding_tokens_per_step,
        "avg_confidence": avg_confidence,
        "dist_decoding_steps": sort_dict(decoding_steps_dist),
        "dist_cache_refresh": sort_dict(cache_refresh_dist),
        "dist_forward_length": sort_dict(forward_length_dist),
        "dist_window_length": sort_dict(window_length_dist),
        "dist_decoding_tokens_per_step": sort_dict(decoding_tokens_per_step_dist),
        "dist_confidence": sort_dict(confidence_dist)
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print required key-value pairs
    print(f"avg_decoding_steps:{avg_decoding_steps}")
    print(f"avg_cache_refresh:{avg_cache_refresh}")
    print(f"avg_forward_length:{avg_forward_length}")
    print(f"avg_window_length:{avg_window_length}")
    print(f"avg_decoding_tokens_per_step:{avg_decoding_tokens_per_step}")
    print(f"avg_confidence:{avg_confidence}")
    print(f"dist_confidence:{sort_dict(confidence_dist)}")


if __name__ == '__main__':
    """
    Test cases for visualization functions
    """
    import torch
    from decode_algorithm import window_bidirectional_decode, BidirectionalDLLM, DecodeConfig
    import os
    
    # Create temporary directory for test outputs
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Test outputs will be saved to: {temp_dir}")
    
    # Test configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate test debug information
    debug_infos_list = []
    
    for run_idx in range(3):
        print(f"\nGenerating test debug info for run {run_idx + 1}")
        
        # Create a simple model and config for testing
        model = BidirectionalDLLM().to(device)
        config = DecodeConfig(
            window_type='sliding',
            generate_length=20,
            initial_window_length=8,
            temperature=0.0,
            decode_algo='threshold',
            decode_param=0.7,
            cache_type='prefix',
            refresh_count=4,
            debug=True,
            mask_id=0,
            pad_id=1,
            eos_id=2
        )
        
        # Generate input tokens
        input_ids = torch.randint(3, 10, (1, 10), device=device)
        
        # Run decoding (this will create debug info)
        try:
            tokens, debug_info = window_bidirectional_decode(model, input_ids, config)
            debug_infos_list.append(debug_info)
            print(f"Run {run_idx + 1}: Generated {len(debug_info)} debug steps")
        except Exception as e:
            print(f"Error in run {run_idx + 1}: {e}")
            # Create mock debug info for testing
            mock_debug_info = []
            for step in range(8 + run_idx):
                mock_step = OneStepDebugInfo(
                    forward_length=10 + step,
                    window_range=(step, step + 8),
                    token_after_decode=torch.randint(3, 10, (1, 10 + step), device=device),
                    token_generated=torch.randint(3, 10, (1, 8), device=device),
                    confidence_generated=torch.rand((1, 8), device=device) * 0.5 + 0.5,
                    decode_indices=[torch.tensor([i for i in range(step, step + 4)], device=device)],
                    decode_confidence=[torch.rand(4, device=device) * 0.5 + 0.5],
                    pass_cache=(step % 4 != 0)  # Refresh cache every 4 steps
                )
                mock_debug_info.append(mock_step)
            debug_infos_list.append(mock_debug_info)
    
    # Test 1: Print decode process for each run
    print("\n" + "="*80)
    print("TEST 1: PRINT DECODE PROCESS")
    print("="*80)
    for i, debug_info in enumerate(debug_infos_list):
        print(f"\n>>> Decode Process for Run {i + 1}:")
        print_decode_process(debug_info)
    
    # Test 2: Line chart for confidence trend
    print("\n" + "="*80)
    print("TEST 2: CONFIDENCE TREND LINE CHARTS")
    print("="*80)
    for i, debug_info in enumerate(debug_infos_list):
        output_file = os.path.join(temp_dir, f'confidence_trend_run_{i+1}.png')
        linechart_confidence_trend(
            debug_info, 
            output_file, 
            show_stages=True,
            figsize=(10, 6),
            dpi=150
        )
        print(f"Generated confidence trend chart for run {i+1}")
    
    # Test 3: Box plot for confidence distribution
    print("\n" + "="*80)
    print("TEST 3: CONFIDENCE DISTRIBUTION BOX PLOT")
    print("="*80)
    output_file = os.path.join(temp_dir, 'confidence_distribution.png')
    boxplot_confidence_distribution(
        debug_infos_list,
        output_file,
        figsize=(12, 8),
        dpi=150,
        max_steps=15
    )
    print("Generated confidence distribution box plot")
    
    # Test 4: Analyze debug infos
    print("\n" + "="*80)
    print("TEST 4: DEBUG INFORMATION ANALYSIS")
    print("="*80)
    output_file = os.path.join(temp_dir, 'debug_analysis.json')
    analyze_debug_infos(debug_infos_list, output_file)
    
    print(f"\nAll test outputs saved to: {temp_dir}")
    print("Visualization tests completed successfully!")