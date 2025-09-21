import pandas as pd
import numpy as np
import os


def safe_divide(numerator, denominator):
    """Realiza divisão segura evitando divisão por zero"""
    return np.where(denominator == 0, 0, numerator / denominator)


def process_chunk(chunk, output_file, is_first_chunk=False):
    """Processa um chunk de dados calculando features derivadas"""
    chunk.columns = ["Protocol", "Flow.Duration", "Total.Fwd.Packets", "Total.Backward.Packets",
                     "Total.Length.of.Bwd.Packets", "Total.Length.of.Fwd.Packets", 
                     "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean", "Average.Packet.Size",
                     "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s", "ProtocolName"]

    numeric_columns = ["Protocol", "Flow.Duration",
                      "Total.Fwd.Packets", "Total.Backward.Packets", "Total.Length.of.Bwd.Packets",
                      "Total.Length.of.Fwd.Packets", "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean",
                      "Average.Packet.Size", "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s"]
    
    for col in numeric_columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)

    chunk["Packets.s"] = safe_divide(chunk["Total.Fwd.Packets"] + chunk["Total.Backward.Packets"], chunk["Flow.Duration"])
    chunk["Fwd.bytes.s"] = safe_divide(chunk["Total.Length.of.Fwd.Packets"], chunk["Flow.Duration"])
    chunk["Bwd.bytes.s"] = safe_divide(chunk["Total.Length.of.Bwd.Packets"], chunk["Flow.Duration"])
    chunk["Fwd.bytes.packet"] = safe_divide(chunk["Total.Length.of.Fwd.Packets"], chunk["Total.Fwd.Packets"])
    chunk["Bwd.bytes.packet"] = safe_divide(chunk["Total.Length.of.Bwd.Packets"], chunk["Total.Backward.Packets"])
    chunk["Speed.packet"] = safe_divide(chunk["Flow.Bytes.s"], chunk["Total.Fwd.Packets"] + chunk["Total.Backward.Packets"])
    chunk["Fwd.Speed.packet"] = safe_divide(chunk["Fwd.bytes.s"], chunk["Total.Fwd.Packets"])
    chunk["Bwd.Speed.packet"] = safe_divide(chunk["Bwd.bytes.s"], chunk["Total.Backward.Packets"])
    chunk["Ratio.Bytes"] = safe_divide(chunk["Total.Length.of.Bwd.Packets"], chunk["Total.Length.of.Fwd.Packets"])
    chunk["Ratio.Packets"] = safe_divide(chunk["Total.Backward.Packets"], chunk["Total.Fwd.Packets"])

    column_order = ["Protocol", "Total.Fwd.Packets", "Total.Backward.Packets", 
                   "Total.Length.of.Bwd.Packets", "Total.Length.of.Fwd.Packets",
                   "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean", "Average.Packet.Size",
                   "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s", "Bwd.bytes.packet",
                   "Fwd.bytes.packet", "Bwd.bytes.s", "Fwd.bytes.s", "Packets.s",
                   "Speed.packet", "Fwd.Speed.packet", "Bwd.Speed.packet",
                   "Ratio.Bytes", "Ratio.Packets", "ProtocolName"]
    
    chunk = chunk.reindex(columns=column_order)
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)
    chunk.to_csv(output_file, mode="a", header=is_first_chunk, index=False, float_format="%.8f")


def main():
    """Função principal para processar o dataset"""
    input_csv_file = "2-processando_dataset/dataset.csv"
    output_csv_file = "2-processando_dataset/dataset_tratado.csv"

    df = pd.read_csv(input_csv_file)
    print(f"Dataset shape: {df.shape}")
    
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)

    chunk_size = 100000
    is_first_chunk = True
    chunk_count = 0

    print(f"\nIniciando processamento em chunks de {chunk_size:,} registros...")
    
    for chunk in pd.read_csv(input_csv_file, chunksize=chunk_size, skiprows=[1], dtype=str, low_memory=False):
        chunk_count += 1
        print(f"Processando chunk {chunk_count}: {len(chunk):,} registros")
        process_chunk(chunk, output_csv_file, is_first_chunk)
        is_first_chunk = False

    print(f"\nProcessamento concluído!")
    print(f"Total de chunks processados: {chunk_count}")
    print(f"Arquivo CSV salvo como: '{output_csv_file}'")


if __name__ == "__main__":
    main()