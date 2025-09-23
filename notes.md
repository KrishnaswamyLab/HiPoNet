- Changed lr and wd to match paper (1e-4) but this gave bad results
- Why is hidden dim 250 instead of 256?
- Is batch size 32 always correct?
- Alpha gradients are going to zero
- Loss just stalls and doesn't go anywhere

Try to set threshold/sigma so you have about ~25% of edges at init (or use elbow method)
sigma=10

Look at HEIST for Cell Clustering

Look at perturbseq data

For unsupervised cell-level embedding
- Use contrastive learning based on spatial + cell type loss
- Use autoencoder with reconstructive loss

Should we try looking at PC.T matrix for point-cloud level embedding? 