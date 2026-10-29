Find the trained LDM (VAE + UNet) and surrogate weights in the `weights/` folder of [https://drive.google.com/drive/folders/1vhaRdzOv56gIDrDwPhm8xDW6S-g_nhXB?usp=sharing](https://drive.google.com/drive/folders/1vhaRdzOv56gIDrDwPhm8xDW6S-g_nhXB?usp=sharing)
(the surrogate can also be retrained with `python scripts/train_surrogate.py configs/surrogate_train.yaml`)

Place them as:

```
weights/
  trained_vae/
    trained_vae.pt
    autoencoder_properties.pkl
  trained_unet/
    trained_unet.pt
    unet_properties.pkl
  trained_surrogate/
    model_trained.pth
    scalers.joblib
```
