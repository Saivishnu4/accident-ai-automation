"""
Training Script for Accident Detection Models
"""
import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime
import sys

import torch
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.damage_classifier import build_damage_classifier_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_yolo(args):
    """Train YOLO object detection model"""
    logger.info("Starting YOLO training...")
    
    # Initialize model
    if args.pretrained:
        model = YOLO('yolov8n.pt')  # Start from pretrained
    else:
        model = YOLO('yolov8n.yaml')  # Start from scratch
    
    # Training arguments
    train_args = {
        'data': args.data_config,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.image_size,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'save': True,
        'project': args.output_dir,
        'name': f'yolo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'lr0': args.learning_rate,
        'weight_decay': args.weight_decay,
        'augment': True,
    }
    
    # Start training
    results = model.train(**train_args)
    
    # Evaluate on validation set
    metrics = model.val()
    
    logger.info(f"Training complete. Results saved to {train_args['project']}")
    logger.info(f"Validation mAP50: {metrics.box.map50:.4f}")
    logger.info(f"Validation mAP50-95: {metrics.box.map:.4f}")
    
    # Export model
    if args.export:
        export_path = Path(args.output_dir) / "best.pt"
        model.export(format='onnx')
        logger.info(f"Model exported to ONNX format")
    
    return results


def train_damage_classifier(args):
    """Train damage severity classifier"""
    logger.info("Starting damage classifier training...")
    
    # Data generators
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Build model
    num_classes = len(train_generator.class_indices)
    model = build_damage_classifier_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes
    )
    
    logger.info(f"Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(Path(args.output_dir) / 'damage_classifier_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(Path(args.output_dir) / 'logs'),
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger(
            str(Path(args.output_dir) / 'training_log.csv')
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'damage_classifier_final.h5'
    model.save(str(final_model_path))
    logger.info(f"Model saved to {final_model_path}")
    
    # Evaluate on validation set
    val_loss, val_acc, val_top2_acc = model.evaluate(val_generator)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Validation Top-2 Accuracy: {val_top2_acc:.4f}")
    
    # Save training history
    import json
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    
    return model, history


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train accident detection models'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolo', 'damage', 'all'],
        help='Model to train'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='data/dataset.yaml',
        help='Path to YOLO data config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/training',
        help='Output directory for models and logs'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Weight decay'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        choices=['Adam', 'SGD', 'AdamW'],
        help='Optimizer'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    # Model arguments
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Export
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export trained model to ONNX'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    return args


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Accident Detection Model Training")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    try:
        if args.model == 'yolo' or args.model == 'all':
            logger.info("\n" + "=" * 80)
            logger.info("Training YOLO Model")
            logger.info("=" * 80)
            train_yolo(args)
        
        if args.model == 'damage' or args.model == 'all':
            logger.info("\n" + "=" * 80)
            logger.info("Training Damage Classifier")
            logger.info("=" * 80)
            train_damage_classifier(args)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
