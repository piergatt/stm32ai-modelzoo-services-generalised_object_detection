# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
import hydra
import argparse
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import mlflow
import tensorflow as tf
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from src.utils import get_config
from src.training import train
from src.evaluation import evaluate
from src.quantization import quantize
from src.prediction import predict
from deployment import deploy, deploy_mpu



# This function turns Tensorflow's eager mode on and off.
# Eager mode is for debugging the Model Zoo code and is slower.
# Do not set argument to True to avoid runtime penalties.
tf.config.run_functions_eagerly(False)


def process_mode(cfg: DictConfig):
    """
    Execution of the various services

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    mode = cfg.operation_mode

    mlflow.log_param("model_path", cfg.general.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    if mode == "training":
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f"[INFO] {len(gpus)} GPUs detected, using MirroredStrategy.")
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                train(cfg)
        else:
            train(cfg)
        print("[INFO] training complete")

    elif mode == "evaluation":
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val(cfg=cfg)
        # Launches evaluation on the target through the model zoo evaluation service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))        
        evaluate(cfg)
        print("[INFO] evaluation complete")

    elif mode == "quantization":
        quantize(cfg)
        print("[INFO] quantization complete")

    elif mode == "prediction":
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val_predict(cfg)
        # Launches prediction on the target through the model zoo prediction service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predict(cfg)
        print("[INFO] prediction complete")

    elif mode == 'benchmarking':
        benchmark(cfg)
        print("[INFO] benchmarking complete")

    elif mode == 'deployment':
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg)
        else:
            deploy(cfg)
        print("[INFO] deployment complete")
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')

    elif mode == 'chain_tqe':
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f"[INFO] {len(gpus)} GPUs detected, using MirroredStrategy.")
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                trained_model_path = train(cfg)
        else:
            trained_model_path = train(cfg)
        quantized_model_path = quantize(cfg, model_path=trained_model_path)
        evaluate(cfg, model_path=quantized_model_path)
        print("Trained model path:", trained_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_tqe complete")

    elif mode == 'chain_tqeb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        trained_model_path = train(cfg)
        quantized_model_path = quantize(cfg, model_path=trained_model_path)
        evaluate(cfg, model_path=quantized_model_path)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Trained model path:", trained_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_tqeb complete")

    elif mode == 'chain_eqe':
        evaluate(cfg)
        quantized_model_path = quantize(cfg)
        evaluate(cfg, model_path=quantized_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_eqe complete")

    elif mode == 'chain_eqeb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        evaluate(cfg)
        quantized_model_path = quantize(cfg)
        evaluate(cfg, model_path=quantized_model_path)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_eqeb complete")

    elif mode == 'chain_qb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        quantized_model_path = quantize(cfg)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_qb complete")

    elif mode == 'chain_qd':
        quantized_model_path = quantize(cfg)
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg, model_path_to_deploy=quantized_model_path)
        else:
            deploy(cfg, model_path_to_deploy=quantized_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_qd complete")

    elif mode == 'prediction':
        predict(cfg)

    else:
        raise RuntimeError(f"Internal error: invalid operation mode: {mode}")

    if mode in ['benchmarking', 'chain_tbqeb', 'chain_qb', 'chain_eqeb']:
        mlflow.log_param("stm32ai_version", cfg.tools.stm32ai.version)
        mlflow.log_param("target", cfg.benchmarking.board)

    # logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='od_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # The default hardware type is "MCU".
    process_mode(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')

    # Add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
