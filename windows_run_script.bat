set CUDA_LAUNCH_BLOCKING=1
set CUDA_VISIBLE_DEVICES=0
python run_classifier.py ^
--task_name CoLA ^
--do_train ^
--do_eval ^
--data_dir glue_data\CoLA ^
--max_seq_length 128 ^
--train_batch_size 32 ^
--eval_batch_size 8 ^
--learning_rate 2e-5 ^
--num_train_epochs 3.0 ^
--output_dir D:\users\v-qianl\GPTComparsion\experiment\CoLA