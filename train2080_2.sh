# CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
# CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_erniernn_ernie2_adaw_10040157.log 2>&1 &
# wait
# echo "ernie-rnn ernie2执行完毕"
# CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie1_adaw_no_kg_10040157.log 2>&1 & 
# CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
# wait
# echo "ernie-rcnn ernie1 和ernir2 执行完毕"

# CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-40-8-2.00E_05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-40-8-2.00E_05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "4号5号执行完毕"
# CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 1e-4 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-20-8-1.00E-04-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 1e-4 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-20-8-1.00E-04-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "6号7号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-20-8-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "8号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-20-8-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "9号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-40-8-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "10号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-40-8-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "11号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-slice-256-0.1-0.1-20-16-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "17号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-slice-256-0.1-0.1-20-16-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "18号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-16-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "19号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-16-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "20号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-slice-256-0.1-0.1-20-32-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "21号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-slice-256-0.1-0.1-20-32-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "22号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
# wait
# echo "23号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "24号执行完毕"


# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "25号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.1e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.100E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "26号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.200E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "27号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.3e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.300E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "28号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.4e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.400E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "29号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.5e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.500E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "30号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.6e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.600E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "31号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.7e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.700E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "32号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.8e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.800E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "33号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 1.9e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-1.900E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "34号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "35号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.1e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.100E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "36号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.2e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.200E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "37号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.3e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.300E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "38号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.4e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.400E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "39号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.5e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.500E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "40号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.6e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.600E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "41号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.7e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.700E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "42号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.8e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.800E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "43号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2.9e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-2.900E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "44号执行完毕"
# python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 3e-5 --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC-256-0.1-0.1-20-32-3.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
# wait
# echo "44号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &

# wait
# echo "55号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_16_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &

# wait
# echo "56号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_16_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &

# wait
# echo "57号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_32_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &

# wait
# echo "58号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_32_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &

# wait
# echo "59号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/93.log 2>&1  &
# wait
# echo "93号执行完毕"

# nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --cuda  > ./outputs/94.log 2>&1  &
# wait
# echo "94号执行完毕"


echo "115号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model bert  --pretrained bert --task MLC --dataset book_multilabels_task --No 115 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/115.log 2>&1  &
wait
echo "115号执行完毕"

echo "118号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model bert-rcnn  --pretrained bert --task MLC --dataset book_multilabels_task --No 118 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/118.log 2>&1  &
wait
echo "118号执行完毕"

echo "119号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie  --pretrained ernie3--task MLC --dataset book_multilabels_task --No 119 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/119.log 2>&1  &
wait
echo "119号执行完毕"

echo "122号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn  --pretrained ernie3--task MLC --dataset book_multilabels_task --No 122 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/122.log 2>&1  &
wait
echo "122号执行完毕"

echo "116号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model bert-rnn  --pretrained bert --task MLC --dataset book_multilabels_task --No 116 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/116.log 2>&1  &
wait
echo "116号执行完毕"

echo "120号开始执行"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rnn  --pretrained ernie3--task MLC --dataset book_multilabels_task --No 120 --learning_rate 2e-5  --seq_length 256 --epochs_num 32 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/120.log 2>&1  &
wait
echo "120号执行完毕"


