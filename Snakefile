rule transform_and_embed:
    input: "{path}/{dataset_name}_embedding_output.parquet"
    output: "{path}/{dataset_name}_umap_annotated.parquet"
    shell: "python scripts/transformation.py --transform {input} {output}"

rule embed:
    input: "{path}/{dataset_name}_embedding_output.parquet"
    output: "{path}/{dataset_name}_umap.parquet",
    shell: "python scripts/transformation.py {input} {output}"

# https://figshare.com/articles/dataset/ISMB_BioVis_2022_Data/20301639
rule download:
    input: "data/figshare/20301639.zip"
    output: directory("data/mair-2022-ismb")
    shell: "unzip {input} -d {output}"

rule figshare_download:
    output: "data/figshare/{article_id}.zip"
    shell: """
    curl -L "https://figshare.com/ndownloader/articles/{wildcards.article_id}/versions/1" -o {output}
    """
