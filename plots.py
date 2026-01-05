#!/dors/meilerlab/data/belle6/miniforge3/envs/glypredreal/bin/python
import re
import argparse
import matplotlib.pyplot as plt

def parse_file(file_path, threshold):
    """
    Parses the log file and extracts epoch data, overall losses/AUPRC and per-CPLM type metrics.
    
    Expected block structure:
      - A tensor block (which may span multiple lines) starting with:
          tensor([
          ... tensor values ...
          ])
      - Followed by one line per CPLM type:
          CPLM/SomeModification.elm.csv <AUPRC> <MCC>
      - And an epoch summary line:
          Epoch 122 Training loss: 71388.764 Validation Loss: 444349.156  Validation AUPRC: 8.62196
    """
    epochs = []
    training_loss = []
    validation_loss = []
    overall_AUPRC = []
    # Dictionary to hold per-type AUPRC and MCC over epochs.
    cplm_data = {}
    # This will hold the names of CPLM types that pass the threshold.
    selected_types = set()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    first_epoch = True
    while i < len(lines):
        line = lines[i].strip()
        # Look for the beginning of a tensor block
        if line.startswith("tensor("):
            # Accumulate all lines belonging to the tensor block
            tensor_block = line
            while not tensor_block.rstrip().endswith("])") and i < len(lines) - 1:
                i += 1
                tensor_block += " " + lines[i].strip()
            # Extract the numbers from the tensor block.
            m_tensor = re.search(r"tensor\(\[([^\]]+)\]\)", tensor_block)
            if m_tensor:
                tensor_str = m_tensor.group(1)
                try:
                    counts = [float(x.strip()) for x in tensor_str.split(',')]
                except Exception as e:
                    print("Error parsing tensor counts:", e)
                    counts = []
            else:
                counts = []
            
            # The number of CPLM lines following should match the number of counts.
            num_types = len(counts)
            cplm_lines = []
            i += 1
            for j in range(num_types):
                if i < len(lines):
                    cplm_lines.append(lines[i].strip())
                    i += 1
            # For the first epoch, decide which CPLM types to track based on the threshold.
            if first_epoch:
                for idx, cline in enumerate(cplm_lines):
                    parts = cline.split()
                    if len(parts) >= 3:
                        type_name = parts[0]  # e.g., "CPLM/Lactylation.elm.csv"
                        # Use the tensor count corresponding to this type.
                        if counts[idx] >= threshold:
                            selected_types.add(type_name)
                            cplm_data[type_name] = {"AUPRC": [], "MCC": []}
                first_epoch = False

            # For each CPLM line in the block, record the values for selected types.
            for idx, cline in enumerate(cplm_lines):
                parts = cline.split()
                if len(parts) >= 3:
                    type_name = parts[0]
                    if type_name in selected_types:
                        try:
                            auprc_val = float(parts[1])
                            mcc_val   = float(parts[2])
                        except Exception as e:
                            print(f"Error converting values for {type_name} at epoch block: {e}")
                            auprc_val, mcc_val = None, None
                        cplm_data[type_name]["AUPRC"].append(auprc_val)
                        cplm_data[type_name]["MCC"].append(mcc_val)
                        
            # Next line should be the epoch summary line.
            if i < len(lines):
                epoch_line = lines[i].strip()
                # Regex to capture epoch number, training loss, validation loss, and overall validation AUPRC.
                m_epoch = re.search(
                    r"Epoch (\d+).*Training loss:\s*([\d\.eE+-]+)\s+Validation Loss:\s*([\d\.eE+-]+)\s+Validation AUPRC:\s*([\d\.eE+-]+)",
                    epoch_line
                )
                if m_epoch:
                    epochs.append(int(m_epoch.group(1)))
                    training_loss.append(float(m_epoch.group(2)))
                    validation_loss.append(float(m_epoch.group(3)))
                    overall_AUPRC.append(float(m_epoch.group(4)))
                else:
                    print("Epoch summary line not found or mismatched:", epoch_line, lines[i+1].strip())
                i += 1
            else:
                i += 1
        else:
            i += 1

    return epochs, training_loss, validation_loss, overall_AUPRC, cplm_data

def plot_data(epochs, training_loss, validation_loss, overall_AUPRC, cplm_data):
    # Plot 1: Training and Validation Loss per Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label="Training Loss", marker='o')
    plt.plot(epochs, validation_loss, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("loss_per_epoch.png")

    # Plot 2: Overall Validation AUPRC vs Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, overall_AUPRC, label="Overall Validation AUPRC", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUPRC")
    plt.title("Overall Validation AUPRC vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("overall_auprc.png")

    # Plot 3: CPLM AUPRC vs Epoch (for types with sufficient data points)
    plt.figure(figsize=(8, 6))
    for type_name, data in cplm_data.items():
        plt.plot(epochs, data["AUPRC"], label=type_name, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("AUPRC")
    plt.title("CPLM AUPRC vs Epoch")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("cplm_auprc.png")

    # Plot 4: CPLM MCC vs Epoch (for types with sufficient data points)
    plt.figure(figsize=(8, 6))
    for type_name, data in cplm_data.items():
        plt.plot(epochs, data["MCC"], label=type_name, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title("CPLM MCC vs Epoch")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("cplm_mcc.png")

    # Optionally display the plots
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Parse training log output and generate plots.")
    parser.add_argument("input_file", type=str, help="Path to the log file with repeated output.")
    parser.add_argument("--threshold", type=float, default=1000.0,
                        help="Minimum number of data points to include a CPLM type in per-type plots (default: 100)")
    args = parser.parse_args()

    epochs, training_loss, validation_loss, overall_AUPRC, cplm_data = parse_file(args.input_file, args.threshold)
    if not epochs:
        print("No epoch data found. Please check the input file format.")
        return
    plot_data(epochs, training_loss, validation_loss, overall_AUPRC, cplm_data)

if __name__ == "__main__":
    main()
