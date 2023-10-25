echo "117号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model bert-cnn  --pretrained bert --task MLC --dataset book_multilabels_task --No 117 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/117.log 2>&1  &
wait
echo "117号执行完毕"

echo "121号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-cnn  --pretrained ernie3--task MLC --dataset book_multilabels_task --No 121 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/121.log 2>&1  &
wait
echo "121号执行完毕"
