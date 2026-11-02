# Image Captioning using ViT + GPT-2

A beginner-friendly image captioning project using **transfer learning** and the **Flickr8k** dataset. The notebook runs on **Kaggle GPU**.

## What this project does

An input photo goes through a **pretrained Vision Transformer (ViT)**, which extracts numerical image features. **GPT-2**, used as a text decoder, learns to attend to those features and generates an English description.

**Architecture:** `Image → ViT encoder (frozen) → GPT-2 decoder + cross-attention (trained) → Caption`

ViT and GPT-2 start with pretrained weights; this project **connects** them and trains the decoder. It does **not** build or train both models from scratch. Unlike an already-caption-trained `nlpconnect/vit-gpt2-image-captioning` checkpoint, the image-to-text cross-attention here is initially untrained.

## Repository contents

```text
vit-gpt2-flickr8k-captioning/
├── notebooks/
│   └── ViT_GPT2_Flickr8k_Kaggle.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Run it on Kaggle

1. Open [Kaggle Code](https://www.kaggle.com/code) and create a **New Notebook**.
2. Upload or import [`notebooks/ViT_GPT2_Flickr8k_Kaggle.ipynb`](notebooks/ViT_GPT2_Flickr8k_Kaggle.ipynb).
3. Select **Add Input → Datasets**, and attach [adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
4. Enable **GPU** and **Internet** (Internet is needed for the first pretrained model download).
5. Run the cells in order.

The code searches `/kaggle/input/` for the folder containing `captions.txt` and `Images/`. You do not need to download Flickr8k yourself or enter a paid API key. Public Hugging Face models can normally be downloaded without a token.

## Training, validation and testing

- **Train:** Updates GPT-2's weights, including its new cross-attention layers.
- **Validation:** Checks loss on separate images without updating weights.
- **Test:** Generates captions on held-out images, computes BLEU against all five references, and displays examples.

The notebook splits by **image**, preventing the same image from being used in both train and test. It uses a reproducible *custom* split, not the standard benchmark split. The default sample counts (`1000/100/100`) are for a small first experiment and should not be presented as published-benchmark results.

## Settings

The notebook has one settings cell:

| Setting | Default | Meaning |
|---|---:|---|
| `TRAIN_SAMPLES` | 1000 | Training images for an initial experiment |
| `VAL_SAMPLES` | 100 | Validation images |
| `TEST_SAMPLES` | 100 | Held-out test images |
| `EPOCHS` | 3 | Number of training passes |
| `BATCH_SIZE` | 1 | Small batch to reduce memory use |
| `GRAD_ACCUM_STEPS` | 4 | Accumulate gradients over several mini-batches |
| `MAX_LENGTH` | 24 | Maximum caption token length |
| `USE_AMP` | GPU only | Mixed precision to help with memory |

For a more serious experiment, increase the training sample count, train longer with checkpoints, and report measured test results. **No performance scores are claimed in this repository** because final experiments have not been verified.

## Troubleshooting

**`GPT2Config` has no `is_decoder`:** The notebook explicitly sets `decoder_config.is_decoder = True` and `decoder_config.add_cross_attention = True` before joining the models.

**`crossattention ... MISSING` on model load:** Expected. Cross-attention layers were not present in GPT-2's original checkpoint and are initialized for training.

**Kernel restarts / GPU memory errors:** Keep `BATCH_SIZE = 1`, leave `USE_AMP` enabled on GPU, restart the session, and first test with fewer images. Increasing batch size generally uses more GPU memory.

**`captions.txt` not found:** Attach the *dataset* `adityajn105/flickr8k`, not a notebook named Flickr8k. You should see an `Images` directory and `captions.txt`.

**Model generates unrelated captions:** The image-text connection is newly initialized. A small training experiment may not learn strong visual grounding. Training on more image-caption examples or using an already-caption-trained model are possible next steps.

## Tools used

Python, PyTorch, Hugging Face Transformers, ViT, GPT-2, Pandas, PIL, SacreBLEU, Matplotlib, Kaggle.

## Dataset and model credit

- [Flickr8k dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [ViT checkpoint](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2)

Dataset images and model weights are **not included** in this repository. Consult the dataset and model pages for their respective terms and licensing.

## Share on GitHub

Create a new **public or private** repository called `vit-gpt2-flickr8k-captioning` on GitHub. Then choose **Add file → Upload files**, and upload the files from this folder while preserving `notebooks/`. Commit your changes. Do not upload any `.env`, API token, Kaggle dataset images, or model checkpoint files.
