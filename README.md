# cev

## setup

```sh
conda env create --file environment.yml
conda activate cev
pip install -e .
```


## release

releases are triggered via tagged commits

```
git tag -a vX.X.X -m "vX.X.X"
git push --follow-tags
```

