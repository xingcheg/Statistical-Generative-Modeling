# Statistical Generative Modeling

**STAT 5095 · University of Connecticut · Fall 2026**

**Instructor:** Xingche Guo

Course slides and Python labs on the statistical foundations of generative modeling, from latent-variable models and posterior inference to variational autoencoders and diffusion models.

> **Course in progress.** Materials currently available in this repository extend through DDPMs. Score-based diffusion models have **not yet been covered**; their materials will be added as the course progresses. Flow matching and modern diffusion models, including Diffusion Transformers (DiT), are **optional topics not yet covered**. The contents below describe the available materials, rather than a complete final syllabus.

## Course overview

The course connects probability models, statistical inference, and neural networks. The central questions are how to describe a data distribution, learn its parameters when latent variables are unobserved, and generate new observations from the fitted model.

The available materials develop:

- Latent-variable modeling, maximum likelihood, and the expectation-maximization (EM) algorithm.
- Hidden Markov models, posterior state inference, and decoding.
- Variational inference, the evidence lower bound (ELBO), and Markov chain Monte Carlo (MCMC).
- Neural-network foundations and practical model training in PyTorch.
- Autoencoders, variational autoencoders (VAEs), and DDPM training and sampling.

Familiarity with probability, statistical inference, linear algebra, multivariable calculus, and basic Python will be helpful. The deep learning overview provides a bridge into the PyTorch labs.

## Lecture materials

| Topic | Slides | Main content |
| --- | --- | --- |
| Introduction | [Lecture 1](introduction/Lecture1.pdf) | Generative modeling, learning and sampling, conditional generation, and latent variables |
| Expectation-maximization | [Lecture 2](EM/Lecture2.pdf) | EM derivation, likelihood ascent, Gaussian mixtures, and extensions |
| Hidden Markov models | [Lecture 3](HMM/Lecture3.pdf) | Forward-backward inference, EM/Baum-Welch estimation, and Viterbi decoding |
| Variational inference | [Lecture 4](Variational%20Inference/Lecture4.pdf) | Posterior approximation, the ELBO, variational EM, amortized and stochastic inference, and variational Bayes |
| Markov chain Monte Carlo | [Lecture 5](MCMC/Lecture5.pdf) | Gibbs sampling, Metropolis-Hastings, posterior summaries, and stationarity |
| Deep learning overview | [DL-1: Foundations](Deep%20Learning%20Overview/Lecture_DL1.pdf) · [DL-1 companion: MLPs in PyTorch](Deep%20Learning%20Overview/Lecture_DL1b.pdf) · [DL-2: CNNs](Deep%20Learning%20Overview/Lecture_DL2.pdf) | MLPs, backpropagation, optimization, CNNs, residual networks, and U-Nets |
| Variational autoencoders | [Lecture 6](VAE/Lecture6.pdf) | Autoencoders, latent representations, reparameterization, the VAE objective, and conditional VAEs |
| Denoising diffusion probabilistic models | [Lecture 7](DDPM/Lecture7.pdf) | Forward noising, learned reverse transitions, noise prediction, training, and generation |

The DL lectures are supporting material for the neural-network labs; study them before the VAE and DDPM implementations if PyTorch is new to you. Lecture 5 is grouped under MCMC, separately from variational inference.

## Notebooks and HTML previews

Each lab has an editable Jupyter notebook and a corresponding HTML export. GitHub can display notebooks directly. To view an HTML export, download it and open it in a browser; the GitHub file view displays its source. Math rendering may require an internet connection.

| Area | Lab | Notebook | HTML |
| --- | --- | --- | --- |
| Deep learning | PyTorch tensors, devices, and autograd | [Notebook](Deep%20Learning%20Overview/00_pytorch_basic.ipynb) | [Preview](Deep%20Learning%20Overview/00_pytorch_basic.html) |
| Deep learning | The PyTorch training workflow | [Notebook](Deep%20Learning%20Overview/01_pytorch_workflow.ipynb) | [Preview](Deep%20Learning%20Overview/01_pytorch_workflow.html) |
| Deep learning | MLPs: circle data and FashionMNIST | [Notebook](Deep%20Learning%20Overview/02_pytorch_mlp1.ipynb) | [Preview](Deep%20Learning%20Overview/02_pytorch_mlp1.html) |
| Deep learning | CIFAR-10: MLP, CNN, AlexNet, VGG16-BN, and ResNet-50 | [Notebook](Deep%20Learning%20Overview/03_pytorch_cnn_cifar10_1.ipynb) | [Preview](Deep%20Learning%20Overview/03_pytorch_cnn_cifar10_1.html) |
| Variational inference | Linear Gaussian decoder, variational EM, and latent recovery | [Notebook](Variational%20Inference/VI_Linear_Gaussian.ipynb) | [Preview](Variational%20Inference/VI_Linear_Gaussian.html) |
| Autoencoders | MNIST: autoencoder versus PCA across latent dimensions | [Notebook](VAE/mnist_autoencoder.ipynb) | [Preview](VAE/mnist_autoencoder.html) |
| VAEs | Generate MNIST digits with a fully connected VAE | [Notebook](VAE/mnist_vae.ipynb) | [Preview](VAE/mnist_vae.html) |
| VAEs | Generate CelebA faces with a convolutional VAE | [Notebook](VAE/celeba_vanilla_vae_lab.ipynb) | [Preview](VAE/celeba_vanilla_vae_lab.html) |
| DDPMs | Generate MNIST digits with a residual U-Net | [Notebook](DDPM/mnist_ddpm_unet.ipynb) | [Preview](DDPM/mnist_ddpm_unet.html) |
| DDPMs | Generate CelebA faces with a residual U-Net | [Notebook](DDPM/celeba_ddpm_unet.ipynb) | [Preview](DDPM/celeba_ddpm_unet.html) |

The autoencoder/PCA lab is a useful bridge to VAEs and is kept in the `VAE/` folder. The DDPM labs use attention-free residual U-Nets; they do not implement DiT or flow matching.

## Topics still to come

| Topic | Course status |
| --- | --- |
| Score-based diffusion models | Upcoming; not yet covered, and materials are not yet included |
| Flow matching | Optional; not yet covered, and materials are not yet included |
| Modern diffusion models, including Diffusion Transformers (DiT) | Optional; not yet covered, and materials are not yet included |

The scope of the optional topics may change as the course progresses.

## Run the labs

Clone the repository and create a Python environment:

```bash
git clone https://github.com/xingcheg/Statistical-Generative-Modeling.git
cd Statistical-Generative-Modeling
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Install the packages used across the labs:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy scipy pandas matplotlib scikit-learn pillow tqdm gdown
python -m pip install torch torchvision
python -m jupyterlab
```

For a GPU-specific PyTorch installation, follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and install a compatible `torch`/`torchvision` pair. Use the Python environment above as the notebook kernel. Some labs also include a setup cell that installs missing packages.

Open a notebook, review its settings cell, and run the cells from top to bottom. The notebooks can also be opened in a Python notebook environment such as Google Colab.

- **Start small.** Where available, set `QUICK_RUN=True`, or `RUN_MODE='demo'` in the CIFAR-10 comparison, to check the workflow before a full training run. These modes use reduced training budgets and should not be treated as final model-quality comparisons.
- **Choose suitable hardware.** The linear Gaussian VI lab and introductory PyTorch examples can run on CPU. A GPU is recommended for the larger CNN, CelebA, and diffusion experiments. Device support varies by notebook; consult its setup cell.
- **Read saved outputs in context.** Outputs show the configuration used when they were produced. Restart the kernel and rerun the notebook after changing settings. Training results and run time can vary with hardware, random seeds, and library versions.
- **Train to produce checkpoints.** The repository does not bundle pretrained weights. Inline figures and saved notebook outputs are distinct from the checkpoint files created during training.

### Data and generated files

The linear Gaussian VI lab simulates its own data. The image labs download MNIST, FashionMNIST, CIFAR-10, or CelebA as needed. Datasets are not bundled in the repository.

For CelebA, read the data-loading section before running the training cells. The VAE and DDPM face labs support either the torchvision loader or an existing aligned-image folder through `DATA_SOURCE='folder'` and `IMAGE_DIR`. If available, provide `list_eval_partition.txt` to preserve the official splits. Automatic downloading can fail because of host quotas; the folder option allows reuse of data already prepared locally.

Paths such as `data/`, `models/`, and `outputs/` are relative to the notebook's working directory. Training may save checkpoints, metric tables, figures, and animations there. These generated folders, Python caches, Jupyter checkpoints, and model-weight files are excluded by `.gitignore`; the shared PDF, notebook, and HTML course materials remain tracked.

## Repository layout

```text
Statistical-Generative-Modeling/
├── introduction/             # Lecture 1
├── EM/                       # Lecture 2
├── HMM/                      # Lecture 3
├── Variational Inference/    # Lecture 4 and the linear Gaussian VI lab
├── MCMC/                     # Lecture 5
├── Deep Learning Overview/   # DL slides and four PyTorch labs
├── VAE/                      # Lecture 6, autoencoder/PCA, and VAE labs
├── DDPM/                     # Lecture 7 and DDPM labs
├── .gitignore
└── README.md
```

## Acknowledgments

Slides and notebooks identify their references and adapted implementations. Please retain those attributions and the third-party license notices included in the materials.
