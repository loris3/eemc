#!/bin/bash

#!/bin/bash

# Get all job IDs for the current user
jobs=$(squeue -u $USER -h -o "%i %j")  # output: JobID JobName

while read -r job_id job_name; do
    if [[ "$job_name" != *"VSCode_Remote_Client"* ]]; then
        scancel "$job_id"
        echo "Canceled job $job_id ($job_name)"
    else
        echo "Skipped job $job_id ($job_name)"
    fi
done <<< "$jobs"

echo "Finished canceling jobs (except VSCode_Remote_Client)."

dirs=("results" "cache" "models" "tables" "figures" "finetuning_eval")

for dir in "${dirs[@]}"; do
    if [ -e "$dir" ]; then
        read -p "Do you want to delete '$dir'? [y/N] " choice
        case "$choice" in
            [yY]|[yY][eE][sS])
                rm -rf "$dir"
                echo "Deleted $dir"
                ;;
            *)
                echo "Skipped $dir"
                ;;
        esac
    else
        echo "$dir does not exist, skipping"
    fi
done

echo "Cleanup complete!"