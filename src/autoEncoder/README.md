### To Train
By default, latent dimention is 128, max epochs is 100
```
python train.py
```
or you want to change the latent dimension to 64 or max epochs to 10
```
python train.py --ld 64 --e 10
```
currently, we support 3 architectures: VAE(default), autoencoder, ResNet
```
python train.py --arc VAE
```

### Evaluate
By default, evaluate latest model following the naming manner 'autoencoder_alldata_{latent_dim}e{num_epochs}.pth' under 'model' folder
```
python evaluate.py
```
currently, we support 3 architectures: VAE(default), autoencoder, ResNet
```
python evaluate.py --arc VAE
```
or you want to alter the model and also provide latent dimension
```
python evaluate.py --model your_model_name.pth --ld 64
```

### To do list
plot the training loss and val loss --done
reconstruct the next time step rather than the current one
Evaluate other architectures: VAE, ResNet based AE --done