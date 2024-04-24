if ! command -v conda --version &> /dev/null
then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O conda.sh
    bash conda.sh -b -p ~/local/miniconda3
    rm -f conda.sh
    ~/local/miniconda3/bin/conda init bash
    . ~/.bashrc
fi