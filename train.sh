#!/usr/bin/env bash
#model_description='bert_base_sequence_level_2-15_94'
#CUDA_VISIBLE_DEVICES=3 python3 text_train.py\
# --model_description $model_description\
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_1-94'
#CUDA_VISIBLE_DEVICES=1 python3 text_train.py\
# --level_list 94 --punctuation_list "." \
# --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
# --model_description $model_description \
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_2-15_94'
#CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
# --level_list 15 94 --punctuation_list "," "." \
# --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
# --model_description $model_description \
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_3-15_32_94'
#CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
# --level_list 15 32 94 --punctuation_list "," ";" "." \
# --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
# --model_description $model_description \
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_1-123'
#CUDA_VISIBLE_DEVICES=1 python3 text_train.py\
# --level_list 123 --punctuation_list "." \
# --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
# --model_description $model_description \
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_2-83_123'
#CUDA_VISIBLE_DEVICES=0 python3 text_train.py\
# --level_list 43 123 --punctuation_list "," "." \
# --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
# --model_description $model_description \
# >> "out/${model_description}.out"

#model_description='bert_base_sequence_level_3-43_83_123'
#CUDA_VISIBLE_DEVICES=0 python3 text_train.py\
# --level_list 43 83 123 --punctuation_list "," ";" "." \
# --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
# --model_description $model_description \
# >> "out/${model_description}.out"


#model_description='bert_base_sequence_level_1-94'
#for i in {1..20}
#do
#   CUDA_VISIBLE_DEVICES=1 python3 text_train.py\
#    --level_list 94 --punctuation_list "." \
#    --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
#    --train_attack \
#     --train_attack_reverse \
#    --model_description $model_description --train_attack_step $i\
# >> "out/${model_description}.out"
#done

#model_description='bert_base_sequence_level_2-15_94'
#for i in {1..20}
#do
#   CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
#    --level_list 15 94 --punctuation_list "," "." \
#    --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
#    --train_attack \
#     --train_attack_reverse \
#    --model_description $model_description --train_attack_step $i\
# >> "out/${model_description}.out"
#done

#model_description='bert_base_sequence_level_3-15_32_94'
#for i in {1..20}
#do
#   CUDA_VISIBLE_DEVICES=3 python3 text_train.py\
#    --level_list 15 32 94 --punctuation_list "," ";" "." \
#    --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
#    --train_attack \
#     --train_attack_reverse \
#    --model_description $model_description --train_attack_step $i\
# >> "out/${model_description}.out"
#done

#model_description='bert_base_sequence_level_1-123'
#for i in {1..20}
#do
#   CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
#    --level_list 123 --punctuation_list "." \
#    --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
#    --train_attack \
#     --train_attack_reverse \
#    --model_description $model_description --train_attack_step $i\
# >> "out/${model_description}.out"
#done

#model_description='bert_base_sequence_level_2-83_123'
#for i in {1..20}
#do
#   CUDA_VISIBLE_DEVICES=4 python3 text_train.py\
#    --level_list 83 123 --punctuation_list "," "." \
#    --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
#    --train_attack \
#     --train_attack_reverse \
#    --model_description $model_description --train_attack_step $i\
# >> "out/${model_description}.out"
#done

model_description='bert_base_sequence_level_3-43_83_123'
for i in {1..20}
do
   CUDA_VISIBLE_DEVICES=6 python3 text_train.py\
    --level_list 43 83 123 --punctuation_list "," ";" "." \
    --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
    --train_attack \
     --train_attack_reverse \
    --model_description $model_description --train_attack_step $i\
 >> "out/${model_description}.out"
done



#model_description='bert_base_sequence_level_2-15_94'
#CUDA_VISIBLE_DEVICES=3 python3 text_train.py\
# --level_list 15 94 --punctuation_list "," "." \
# --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" ----log_dir "log_20" \
# --train_attack "False" --train_attack_reverse "True" \
# --model_description $model_description\
# >> "out/${model_description}.out"


#model_description='bert_base_sequence_level_3-15_32_94'
#CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
# --level_list 15 32 94 --punctuation_list "," ";" "." \
# --train_dataset "ADReSS20-train" --test_dataset "ADReSS20-test" --log_dir "log_20" \
# --model_description $model_description \
# >> "out/${model_description}.out"


#model_description='bert_base_sequence_level_3-43_83_123'
#CUDA_VISIBLE_DEVICES=2 python3 text_train.py\
# --level_list 43 83 123 --punctuation_list "," ";" "." \
# --train_dataset "ADReSSo21-train" --test_dataset "ADReSSo21-test" --log_dir "log_21" \
# --model_description $model_description \
# >> "out/${model_description}.out"
