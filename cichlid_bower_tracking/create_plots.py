import pandas as pd
import matplotlib.pyplot as plt
import math,os
# Load the CSV file
def plot_behavior_scatter(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file)
    image_height = 972
    image_width = 1296
    # Ensure columns exist
    required_columns = {'VideoID', 'X', 'Y', 'predicted_label'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    # Filter behaviors that contain 'c', 's', 'p', or 'b'
    # behaviors_of_interest = ['c', 's', 'p', 'b']
    # behaviors_of_interest = ['t']
     # Define behaviors of interest in the desired order
    behaviors_of_interest = ['c','p','b','f','t','m','s']
    df_filtered = df[df['predicted_label'].isin(behaviors_of_interest)]
    
    # Assign consistent colors based on behavior order
    colors = plt.cm.get_cmap('tab10', len(behaviors_of_interest))
    color_map = {behavior: colors(i) for i, behavior in enumerate(behaviors_of_interest)}

    # Get unique videos
    unique_videos = df_filtered['VideoID'].unique()
    output_folder = "/home/pkolipaka3@ad.gatech.edu/Temp/CichlidAnalyzer/__ProjectData/MC_multi/MC_s7_tr2_BowerBuilding/Output_plots"
    os.makedirs(output_folder,exist_ok=True)
    # num_videos = len(unique_videos)
    # num_behaviors = len(behaviors_of_interest)


    legend_labels = { "c": "Build Scoop", "s": "Quiver/Spawn", "p": "Build Spit", "b": "Build Multiple","f": "Feed Scoop", "t": "Feed Spit", "m": "Feed Multiple" }

    # fig, axes = plt.subplots(num_behaviors, num_videos, figsize=(5 * num_videos, 5 * num_behaviors), sharex=True, sharey=True)
    # if num_behaviors == 1:
    #     axes = [axes]  # Ensure axes is iterable if only one row
    # if num_videos == 1:
    #     axes = [[ax] for ax in axes]  # Ensure axes is a 2D array if only one column
    
    # for col, video in enumerate(unique_videos):
    #     df_video = df_filtered[df_filtered['VideoID'] == video]
        
    #      # Add video titles on top of the first row
    #     axes[0][col].set_title(video, fontsize=12)
        
    #     for row, behavior in enumerate(behaviors_of_interest):
    #         ax = axes[row][col]
    #         subset = df_video[df_video['predicted_label'] == behavior]
    #         ax.scatter(subset['Y'], subset['X'], color=color_map.get(behavior, 'black'), alpha=0.7)
            
    #         if col == 0:
    #             ax.set_ylabel(f"{legend_labels.get(behavior, behavior)}\nY Coordinate")
    #         if row == num_behaviors - 1:
    #             ax.set_xlabel(f"{video}\nX Coordinate")
            
    #         ax.set_xlim([0, image_width])
    #         ax.set_ylim([0, image_height])
    #         ax.invert_xaxis()
    #         ax.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, f"{video}.png"))
    # plt.close()
    for video in unique_videos:
        df_video = df_filtered[df_filtered['VideoID'] == video]
        
        plt.figure(figsize=(8, 6))
        
        for behavior in behaviors_of_interest:
            subset = df_video[df_video['predicted_label'] == behavior]
            plt.scatter(subset['Y'], subset['X'], label=legend_labels.get(behavior, behavior), 
                        color=color_map.get(behavior, 'black'), alpha=0.7)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"Behavior Scatter Plot - {video}")
        plt.xlim([0, image_width])
        plt.ylim([0, image_height])
        plt.legend(title="Behavior")
        plt.gca().invert_xaxis()  # Adjust orientation if needed
        plt.grid(True)
        
        # Save plot
        output_file = os.path.join(output_folder, f"1{video}.png")
        plt.savefig(output_file)
        plt.close()
    # # plt.gca().invert_yaxis() # Invert the x axis
    # plt.tight_layout()
    # plt.show()
# Example usage
# plot_behavior_scatter("your_file.csv")
plot_behavior_scatter("/home/pkolipaka3@ad.gatech.edu/Temp/CichlidAnalyzer/__ProjectData/MC_multi/MC_s7_tr2_BowerBuilding/MasterAnalysisFiles/AllLabeledClusters.csv")