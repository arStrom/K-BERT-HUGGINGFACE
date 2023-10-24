python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstm  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstm_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &
wait
echo "54执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &
wait
echo "60执行完毕"

python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &
wait
echo "61执行完毕"

python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_16_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &
wait
echo "62执行完毕"

python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 16 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_16_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &
wait
echo "63执行完毕"

python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --no_kg --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_TRUE_TRUE_AdamW.log 2>&1  &
wait
echo "64执行完毕"

python3 -u run_kbert_transformers_cls.py --model ernie-rcnn-catlstmwide  --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice  --seq_length 256 --epochs_num 20 --batch_size 32 --kg_name CnDbpedia --cuda  > ./outputs/ernie-rcnn-catlstmwide_ernie1_book_multilabels_task_slice_MLC-slice_256_0.1_0.1_20_8_2.00E-05_100_CnDbpedia_FALSE_FALSE_AdamW.log 2>&1  &
wait
echo "65执行完毕"
