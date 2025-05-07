import pandas as pd


def result_analysis():
    tasks = ['fb', 'osmc', 'books']  # Only 'fb' task
    indexs = ['DynamicPGM', 'LIPP', 'HybridPGMLIPPASYNC']  # Added HybridPGMLIPP
    # Create dictionaries to store throughput data for each index
    # lookuponly_throughput = {}
    # insertlookup_throughput = {}
    insertlookup_mix1_throughput = {}
    insertlookup_mix2_throughput = {}
    
    # Store the best hyperparameters for each index and task
    best_hyperparams_mix1 = {}
    best_hyperparams_mix2 = {}
    
    # Store index sizes
    index_sizes_mix1 = {}
    index_sizes_mix2 = {}
    
    for index in indexs:
        # lookuponly_throughput[index] = {}
        # insertlookup_throughput[index] = {"lookup": {}, "insert": {}}
        insertlookup_mix1_throughput[index] = {}
        insertlookup_mix2_throughput[index] = {}
        best_hyperparams_mix1[index] = {}
        best_hyperparams_mix2[index] = {}
        index_sizes_mix1[index] = {}
        index_sizes_mix2[index] = {}
    
    for task in tasks:
        full_task_name = f"{task}_100M_public_uint64"
        # lookup_only_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.000000i_results_table.csv")
        # insert_lookup_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.500000i_0m_results_table.csv")
        insert_lookup_mix_1_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.100000i_0m_mix_results_table.csv", engine='python', on_bad_lines='skip')
        insert_lookup_mix_2_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.900000i_0m_mix_results_table.csv",
                                               engine='python', on_bad_lines='skip')
        
        for index in indexs:
            # find the row where lookup_only_result['index_name'] == index
            # try:
            #     lookup_only_result = lookup_only_results[lookup_only_results['index_name'] == index]
            #     # compute average throughput across lookup_only_result['throughput1'], lookup_only_result['throughput2'], lookup_only_result['throughput3'], then select the one with the highest throughput
            #     lookuponly_throughput[index][task] = round(lookup_only_result[['lookup_throughput_mops1', 'lookup_throughput_mops2', 'lookup_throughput_mops3']].mean(axis=1).max(), 4)
            # except:
            #     pass
            
            # find the row where insert_lookup_result['index_name'] == index
            # try:
            #     insert_lookup_result = insert_lookup_results[insert_lookup_results['index_name'] == index]
            #     # compute average throughput across insert_lookup_result['throughput1'], insert_lookup_result['throughput2'], insert_lookup_result['throughput3'], then select the one with the highest throughput
            #     insertlookup_throughput[index]['lookup'][task] = round(insert_lookup_result[['lookup_throughput_mops1', 'lookup_throughput_mops2', 'lookup_throughput_mops3']].mean(axis=1).max(), 4)
            #     insertlookup_throughput[index]['insert'][task] = round(insert_lookup_result[['insert_throughput_mops1', 'insert_throughput_mops2', 'insert_throughput_mops3']].mean(axis=1).max(), 4)
            # except:
            #     pass
            
                
            # find the row where insert_lookup_mix_1_result['index_name'] == index
            insert_lookup_mix_1_result = insert_lookup_mix_1_results[insert_lookup_mix_1_results['index_name'] == index]
            # compute average throughput for each row
            insert_lookup_mix_1_result['avg_throughput'] = insert_lookup_mix_1_result[['mixed_throughput_mops1', 'mixed_throughput_mops2', 'mixed_throughput_mops3']].mean(axis=1)
            # select the row with the highest average throughput
            try:
                best_row = insert_lookup_mix_1_result.loc[insert_lookup_mix_1_result['avg_throughput'].idxmax()]
                insertlookup_mix1_throughput[index][task] = round(best_row['avg_throughput'], 4)
            except:
                print(f"No best row found for {index} {task}")
                insertlookup_mix1_throughput[index][task] = 0
                # import pdb; pdb.set_trace()
            # Store the index size for the best configuration
            index_sizes_mix1[index][task] = best_row['index_size_bytes']
            
            # Store the best hyperparameters
            hyperparams = {}
            if 'search_method' in best_row and not pd.isna(best_row['search_method']):
                hyperparams['search_method'] = best_row['search_method']
            if 'value' in best_row and not pd.isna(best_row['value']):
                hyperparams['value'] = int(best_row['value'])
            if 'flush_threshold' in best_row and not pd.isna(best_row['flush_threshold']):
                hyperparams['flush_threshold'] = best_row['flush_threshold']
            best_hyperparams_mix1[index][task] = hyperparams
            
            
            # find the row where insert_lookup_mix_2_result['index_name'] == index
            insert_lookup_mix_2_result = insert_lookup_mix_2_results[insert_lookup_mix_2_results['index_name'] == index]
            # compute average throughput for each row
            insert_lookup_mix_2_result['avg_throughput'] = insert_lookup_mix_2_result[['mixed_throughput_mops1', 'mixed_throughput_mops2', 'mixed_throughput_mops3']].mean(axis=1)
            try:
                # select the row with the highest average throughput
                best_row = insert_lookup_mix_2_result.loc[insert_lookup_mix_2_result['avg_throughput'].idxmax()]
                insertlookup_mix2_throughput[index][task] = round(best_row['avg_throughput'], 4)
            except:
                print(f"No best row found for {index} {task}")
                insertlookup_mix2_throughput[index][task] = 0
                # import pdb; pdb.set_trace()
            
            # Store the index size for the best configuration
            if index not in index_sizes_mix2:
                index_sizes_mix2[index] = {}
            index_sizes_mix2[index][task] = best_row['index_size_bytes']
            
            # Store the best hyperparameters
            hyperparams = {}
            if 'search_method' in best_row and not pd.isna(best_row['search_method']):
                hyperparams['search_method'] = best_row['search_method']
            if 'value' in best_row and not pd.isna(best_row['value']):
                hyperparams['value'] = int(best_row['value'])
            if 'flush_threshold' in best_row and not pd.isna(best_row['flush_threshold']):
                hyperparams['flush_threshold'] = best_row['flush_threshold']
            best_hyperparams_mix2[index][task] = hyperparams
    
    # Save data to CSV files for further analysis
    import os
    os.makedirs('analysis_results', exist_ok=True)
    
    # pd.DataFrame(lookuponly_throughput).to_csv('analysis_results/lookuponly_throughput.csv')
    
    # lookup_df = pd.DataFrame({idx: data['lookup'] for idx, data in insertlookup_throughput.items()})
    # insert_df = pd.DataFrame({idx: round(data['insert'], 4) for idx, data in insertlookup_throughput.items()})
    # lookup_df.to_csv('analysis_results/insertlookup_lookup_throughput.csv')
    # insert_df.to_csv('analysis_results/insertlookup_insert_throughput.csv')
    
    pd.DataFrame(insertlookup_mix1_throughput).to_csv('analysis_results/insertlookup_mix1_throughput.csv')
    pd.DataFrame(insertlookup_mix2_throughput).to_csv('analysis_results/insertlookup_mix2_throughput.csv')
    pd.DataFrame(index_sizes_mix1).to_csv('analysis_results/index_sizes_mix1.csv')
    pd.DataFrame(index_sizes_mix2).to_csv('analysis_results/index_sizes_mix2.csv')
    
    # plot the figure of throughput, x axis is the index, y axis is the throughput
    # the figure should contain 4 subplots, each subplot corresponds to a workload, including lookup_only, insert_lookup, insert_lookup_mix1, insert_lookup_mix2
    # each subplot should contain 3 bars, each bar corresponds to a dataset (fb, osmc, books) if the throughput is not empty
    
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Flatten axs for easier indexing
    axs = axs.flatten()
    
    # Define common plot parameters
    bar_width = 0.2
    index = range(len(indexs))
    colors = ['blue', 'green', 'red', 'orange']
    # colors = ['blue', 'green']
    
    # 1. Plot lookup-only throughput
    # ax = axs[0]
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(lookuponly_throughput[idx].get(task, 0))
    #     ax.bar([x + i*bar_width for x in index], task_data, bar_width, label=task, color=colors[i])
        
    # ax.set_title('Lookup-only Throughput')
    # ax.set_ylabel('Throughput (Mops/s)')
    # ax.set_xticks([x + bar_width*1.5 for x in index])
    # ax.set_xticklabels(indexs)
    # ax.legend()
    
    # 2. Plot insert-lookup throughput (separated)
    # ax = axs[1]
    # First plot lookups
    # offset = 0
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(insertlookup_throughput[idx]['lookup'].get(task, 0))
    #     ax.bar([x + offset for x in index], task_data, bar_width/2, 
    #            label=f'{task} (lookup)' if offset == 0 else "_nolegend_", 
    #            color=colors[i])
    #     offset += bar_width/2
    
    # Then plot inserts
    # offset = bar_width*2
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(insertlookup_throughput[idx]['insert'].get(task, 0))
    #     ax.bar([x + offset for x in index], task_data, bar_width/2, 
    #            label=f'{task} (insert)', color=colors[i], hatch='///')
    #     offset += bar_width/2
    
    # ax.set_title('Insert-Lookup Throughput (50% insert ratio)')
    # ax.set_ylabel('Throughput (Mops/s)')
    # ax.set_xticks([x + bar_width*1.5 for x in index])
    # ax.set_xticklabels(indexs)
    # ax.legend()
    
    # 3. Plot mixed workload with 10% inserts
    ax = axs[0]
    for i, task in enumerate(tasks):
        task_data = []
        bar_positions = []
        for j, idx in enumerate(indexs):
            throughput = insertlookup_mix1_throughput[idx].get(task, 0)
            task_data.append(throughput)
            bar_positions.append(j + i*bar_width)
            
            # Format and add hyperparameter labels
            if idx in best_hyperparams_mix1 and task in best_hyperparams_mix1[idx]:
                params = best_hyperparams_mix1[idx][task]
                label_text = ""
                if 'search_method' in params:
                    label_text += f"{params['search_method']}"
                if 'value' in params:
                    label_text += f"\nv={params['value']}"
                if 'flush_threshold' in params:
                    label_text += f"\nFT={params['flush_threshold'].split('=')[-1]}"
                
                if label_text:
                    ax.text(j + i*bar_width, throughput + 0.1, label_text, 
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
        ax.bar(bar_positions, task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Mixed Workload (10% insert ratio)')
    ax.set_ylabel('Throughput (Mops/s)')
    ax.set_xticks([x for x in index])
    ax.set_xticklabels(indexs, rotation=15)
    ax.legend()
    
    # 4. Plot mixed workload with 90% inserts
    ax = axs[1]
    for i, task in enumerate(tasks):
        task_data = []
        bar_positions = []
        for j, idx in enumerate(indexs):
            throughput = insertlookup_mix2_throughput[idx].get(task, 0)
            task_data.append(throughput)
            bar_positions.append(j + i*bar_width)
            
            # Format and add hyperparameter labels
            if idx in best_hyperparams_mix2 and task in best_hyperparams_mix2[idx]:
                params = best_hyperparams_mix2[idx][task]
                label_text = ""
                if 'search_method' in params:
                    label_text += f"{params['search_method']}"
                if 'value' in params:
                    label_text += f"\nv={params['value']}"
                if 'flush_threshold' in params:
                    label_text += f"\nFT={params['flush_threshold'].split('=')[-1]}"
                
                if label_text:
                    ax.text(j + i*bar_width, throughput + 0.1, label_text, 
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
        ax.bar(bar_positions, task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Mixed Workload (90% insert ratio)')
    ax.set_ylabel('Throughput (Mops/s)')
    ax.set_xticks([x for x in index])
    ax.set_xticklabels(indexs, rotation=15)
    ax.legend()
    
    # 5. Plot index size for 10% insert ratio
    ax = axs[2]
    for i, task in enumerate(tasks):
        task_data = []
        bar_positions = []
        for j, idx in enumerate(indexs):
            if task in index_sizes_mix1.get(idx, {}):
                # Convert bytes to MB for better readability
                size_mb = index_sizes_mix1[idx][task] / (1024 * 1024)
                task_data.append(size_mb)
                bar_positions.append(j + i*bar_width)
                
                # Add size labels
                ax.text(j + i*bar_width, size_mb + 100, f"{size_mb:.0f} MB", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
            else:
                task_data.append(0)
                bar_positions.append(j + i*bar_width)
                
        ax.bar(bar_positions, task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Index Size (10% insert ratio)')
    ax.set_ylabel('Size (MB)')
    ax.set_xticks([x for x in index])
    ax.set_xticklabels(indexs, rotation=15)
    ax.legend()
    
    # 6. Plot index size for 90% insert ratio
    ax = axs[3]
    for i, task in enumerate(tasks):
        task_data = []
        bar_positions = []
        for j, idx in enumerate(indexs):
            if task in index_sizes_mix2.get(idx, {}):
                # Convert bytes to MB for better readability
                size_mb = index_sizes_mix2[idx][task] / (1024 * 1024)
                task_data.append(size_mb)
                bar_positions.append(j + i*bar_width)
                
                # Add size labels
                ax.text(j + i*bar_width, size_mb + 100, f"{size_mb:.0f} MB", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
            else:
                task_data.append(0)
                bar_positions.append(j + i*bar_width)
                
        ax.bar(bar_positions, task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Index Size (90% insert ratio)')
    ax.set_ylabel('Size (MB)')
    ax.set_xticks([x for x in index])
    ax.set_xticklabels(indexs, rotation=15)
    ax.legend()
    
    # Add overall title and adjust layout
    fig.suptitle('Benchmark Results Across Different Workloads', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()
    

if __name__ == "__main__":
    result_analysis()