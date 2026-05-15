#!/bin/bash
SKIP_DEPS=true
source .env
EVALDIR="../finetuning_eval"

cd "$CONTAINER_WORKDIR/olmes" || { echo "Clone olmes first!"; exit 1; }


tasks_code=(
    olmo3:base_easy:code_bpb
)

tasks_math=(
    olmo3:base_easy:math_bpb 
)

tasks_sft=(
    mmlu:mc::olmes
    olmo_2_generative::olmes
    olmo_2_heldout::olmes
)
    





readarray -t models < <(
python3 - << 'EOF'
import yaml

with open("../experiments.yaml") as f:
    cfg = yaml.safe_load(f)

for base in cfg["base_models"]:
    for exp in cfg["experiments"]:
        base_model = base.split("/")[1]
        model_dir = f"./models/{base_model}{exp['model']}"

        print(f"{base} {model_dir} {exp['name']}")
EOF
)




baseline_models=(
    # "allenai/OLMo-2-0425-1B-SFT"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
)


merge_list=()
for entry in "${models[@]}"; do
    base_model=$(echo "$entry" | cut -d' ' -f1)
    adapter_model=$(echo "$entry" | cut -d' ' -f2)
    merged_model_dir="../${adapter_model}-merged"
    merge_list+=("$base_model|$adapter_model|$merged_model_dir")
done

export MERGE_LIST="$(printf "%s\n" "${merge_list[@]}")"

merge_array_length=$(echo "$MERGE_LIST" | wc -l)
merge_script=$(mktemp)
cat > "$merge_script" <<EOL
#!/bin/bash
#SBATCH --job-name="[EEMC] Merge"
#SBATCH --container-image="$CONTAINER_IMAGE"
#SBATCH --nodelist=galadriel,shelob,dgx-h100-em2,vader
#SBATCH --container-mount-home
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=0-01:00:00
# #SBATCH --container-workdir=$CONTAINER_WORKDIR
#SBATCH --array=0-$(($merge_array_length - 1))%4
source .env
export HF_HUB_ENABLE_HF_TRANSFER=0

cd $CONTAINER_WORKDIR/olmes
if ! grep -qxF 'import multiprocessing as mp' oe_eval/run_eval.py; then
    sed -i '/if __name__ == "__main__":/a import multiprocessing as mp\nmp.set_start_method("spawn", force=True)' oe_eval/run_eval.py
fi

IFS=\$'\n' read -r -d '' -a items <<< "\$MERGE_LIST"
IFS='|' read -r base_model adapter_model merged_model_dir <<< "\${items[\$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "\$merged_model_dir" ]; then
    echo "Merging \$adapter_model"
    python3 ../merge_lora_to_hf.py \
        --base_model "\$base_model" \
        --adapter_dir "../\${adapter_model}" \
        --output_dir "\$merged_model_dir"
else
    echo "Already merged: \$merged_model_dir"
fi
EOL

if [[ "$SKIP_DEPS" == true ]]; then
    echo "Skipping merge job submission"
    merge_jid=""
else
    merge_jid=$(sbatch "$merge_script" | awk '{print $4}')
    echo "Submitted merge job: $merge_jid"
fi
rm "$merge_script"


eval_list=()

for entry in "${models[@]}"; do
    base_model=$(echo "$entry" | awk '{print $1}')
    adapter_model=$(echo "$entry" | awk '{print $2}')
    exp_name=$(echo "$entry" | awk '{print $3}')

    merged_model_dir="../${adapter_model}-merged"

    # Skip models that don't contain sft/code/math in their name
    if [[ "$adapter_model" != *sft* && \
        "$adapter_model" != *code* && \
        "$adapter_model" != *math* ]]; then
        # echo "Skipping $adapter_model (does not contain sft/code/math)"
        continue
    fi

    # Select task subset based on experiment type
    case "$exp_name" in
        code)
            selected_tasks=("${tasks_code[@]}")
            ;;
        math)
            selected_tasks=("${tasks_math[@]}")
            ;;
        sft)
            selected_tasks=("${tasks_sft[@]}")
            ;;
        *)
            # echo "Skipping $adapter_model (no matching task group for experiment: $exp_name)"
            continue
            ;;
    esac
    for task in "${selected_tasks[@]}"; do
        out_dir="$EVALDIR/$task/${adapter_model}-merged"
        eval_list+=("$task|$merged_model_dir|$out_dir")
    done
done
# exit()
export EVAL_LIST="$(printf "%s\n" "${eval_list[@]}")"

eval_array_length=$(echo "$EVAL_LIST" | wc -l)
eval_script=$(mktemp)
cat > "$eval_script" <<EOL
#!/bin/bash
#SBATCH --job-name="[EEMC] Olmes eval"
#SBATCH --nodelist=galadriel,shelob,dgx-h100-em2,vader
#SBATCH --container-mount-home
#SBATCH --container-image="$CONTAINER_IMAGE_EVAL"
#SBATCH --requeue
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=p_datamining,p_csunivie_gres
#SBATCH --account=datamining
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --container-workdir=$CONTAINER_WORKDIR
#SBATCH --array=0-$(($eval_array_length - 1))%12

source .env
export HF_HUB_ENABLE_HF_TRANSFER=0

# pip install lm-eval==0.4.3

cd $CONTAINER_WORKDIR/olmes

IFS=\$'\n' read -r -d '' -a items <<< "\$EVAL_LIST"
IFS='|' read -r task model out_dir <<< "\${items[\$SLURM_ARRAY_TASK_ID]}"
model_name=\$(basename "\$model")
echo "Evaluating task=\$task model=\$model out_dir=\$out_dir model_name=\$model_name "



pip freeze
python3 -m oe_eval.launch \
    --model "\$model" \
    --num-workers=1 \
    --gpus=1 \
    --task "\$task" \
    --output-dir "\$out_dir"


EOL


if [[ "$SKIP_DEPS" == true ]]; then
    echo "Submitting eval job WITHOUT dependency"
    sbatch "$eval_script"
else
    echo "Submitting eval job WITH dependency on merge job $merge_jid"
    sbatch --dependency=afterany:$merge_jid "$eval_script"
fi




exit()




eval_list=()

for entry in "${models[@]}"; do
    read -r base_model adapter_model exp_name <<< "$entry"

    # Only run custom evals for ftrace + squad models
    if [[ "$exp_name" != "ftrace" && "$exp_name" != "squad_v2" ]]; then
        # echo "Skipping $adapter_model for custom evals (exp=$exp_name)"
        continue
    fi

    merged_model_dir="./${adapter_model}"
    out_dir="$EVALDIR/${adapter_model}"



    eval_list+=("$merged_model_dir|$out_dir|$exp_name")
done

for base_model in "${baseline_models[@]}"; do
    out_dir="$EVALDIR/${base_model}"
    eval_list+=("$base_model|$out_dir|base")
done

export EVAL_LIST="$(printf "%s\n" "${eval_list[@]}")"

eval_array_length=$(echo "$EVAL_LIST" | wc -l)
eval_script=$(mktemp)
cat > "$eval_script" <<EOL
#!/bin/bash
#SBATCH --job-name="[EEMC] Custom eval"
#SBATCH --nodelist=galadriel,shelob,dgx-h100-em2,vader
#SBATCH --container-mount-home
#SBATCH --container-image="$CONTAINER_IMAGE_EVAL"
#SBATCH --partition=p_datamining,p_csunivie_gres
#SBATCH --account=datamining
#SBATCH --requeue
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --container-workdir=$CONTAINER_WORKDIR
#SBATCH --array=0-$(($eval_array_length - 1))%12

source .env
export HF_HUB_ENABLE_HF_TRANSFER=0


IFS=\$'\n' read -r -d '' -a items <<< "\$EVAL_LIST"
IFS='|' read -r model out_dir exp_name <<< "\${items[\$SLURM_ARRAY_TASK_ID]}"
model_name=\$(basename "\$model")
echo "Evaluating model=\$model out_dir=\$out_dir model_name=\$model_name on custom tasks "



if [[ "\$exp_name" == "ftrace" ]]; then
    python3 qa_eval.py \
        --model_path "\$model" \
        --dataset loris3/ftrace_date-lm_conversational \
        --split train

elif [[ "\$exp_name" == "squad_v2" ]]; then
    python3 qa_eval.py \
        --model_path "\$model" \
        --dataset loris3/squad_v2_conversational \
        --split validation

elif [[ "\$exp_name" == "base" ]]; then
    # baselines run both
    python3 qa_eval.py \
        --model_path "\$model" \
        --dataset loris3/ftrace_date-lm_conversational \
        --split train

    python3 qa_eval.py \
        --model_path "\$model" \
        --dataset loris3/squad_v2_conversational \
        --split validation
fi

EOL


if [[ "$SKIP_DEPS" == true ]]; then
    echo "Submitting custom eval WITHOUT dependency"
    sbatch "$eval_script"
else
    echo "Submitting custom eval WITH dependency on merge job $merge_jid"
    sbatch --dependency=afterany:$merge_jid "$eval_script"
fi