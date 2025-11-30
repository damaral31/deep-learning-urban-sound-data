# Classificação Multi-Classe de Áudio usando Deep Learning no dataset UrbanSound8K

## Como executar o projeto

### Instalar ambiente

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/deep-learning-urban-sound-data.git
    cd deep-learning-urban-sound-data
    ```

2. Instale o ambiente Conda usando o arquivo `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate urban-sound-env
    ```

### Instalar dataset:

Para o dataloader funcionar, é necessário downloadar o dataset UrbanSound8k pela sua biblioteca nativa. Para issoÇ

1. Execute o ficheiro:
    ```bash
    cd utils/download_dataset.py
    ```

### Instalar cache:

Para acelearar o processo de treino, é necessário preprocesssar o dataset completo. Para isso, execute o seguinte ficheiro exclusivamente após instalar o dataset:

1. Execute o ficheiro para obter o dataset preprocessado:
    ```bash
    python store_preprocessed_dataset.py
    ```

2. Execute o ficheiro para obter o dataset preprocessado com augmentação:
    ```bash
    python store_preprocessed_augmentation.py
    ```

### Treinar um modelo:

Para treinar um modelo, deve seguir o seguinte esqueleto:

    ```python
    from models.CNN import SoundCNN
    import os

    FOLDS = [f"fold{i}" for i in range(1, 11)]
    DATA_PATH = "datasets"

    model_instance = SoundCNN(num_classes=10, SqueezeExcitation=False)
    trainer = Train(dataset_root_path=DATA_PATH, name="CNN", dataset_type="singlechannel", model=model_instance, num_classes=10, batch_size=64, epochs=50, learning_rate=1e-3, patience=5)
    trainer.run(folds=FOLDS)

    '''

    dataset_root_path -> onde instalou o modelo
    name -> nome do modelo
    dataset_type -> o tipo de dataset. pode ser ["preprocessing", "augmentation_preprocessing", "singlechannel", "augmentation_singlechannel"]
    model -> a instância do modelo a treinar

    '''

    ```