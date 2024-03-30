# DATR: Domain Agnostic Text Recognizer

## Training
```python train.py trainer.gpus=1 ckpt_name=datr_real dataset=real model=datr model.batch_size=320 trainer.val_check_interval=1.0 trainer.max_epochs=5 model.lr=0.0014```

## Dataset Structure
```
data/
    train/
        real/
            IIIT5k/
                data.mdb
                lock.mdb
            SVT/
                data.mdb
                lock.mdb
            ...
    val/
        IC13/
            data.mdb
            lock.mdb
        ...
    test/
        IC15/
            data.mdb
            lock.mdb
        SVT/
            data.mdb
            lock.mdb
        ...
```