from Bio import SeqIO
from Bio.Seq import Seq
import csv

# 替换为你的 GenBank 文件路径
genbank_file = "your_file.gb"
output_csv = "cds_features.csv"

with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['gene', 'product', 'nucleotide_sequence', 'protein_sequence'])

    for record in SeqIO.parse(genbank_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                gene = feature.qualifiers.get("gene", [""])[0]
                product = feature.qualifiers.get("product", [""])[0]

                # 获取核酸序列（考虑互补链）
                location = feature.location
                nucleotide_seq = location.extract(record.seq)
                if location.strand == -1:
                    nucleotide_seq = nucleotide_seq.reverse_complement()

                # 获取蛋白质序列
                if "translation" in feature.qualifiers:
                    protein_seq = feature.qualifiers["translation"][0]
                else:
                    # 如果没有 translation，尝试翻译
                    try:
                        protein_seq = nucleotide_seq.translate(to_stop=True)
                    except Exception as e:
                        protein_seq = f"Translation error: {e}"

                writer.writerow([gene, product, str(nucleotide_seq), str(protein_seq)])
