import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def categorize_protocol(protocol_name):
    """Categoriza protocolos em grupos funcionais"""
    streaming_protocols = ["SPOTIFY", "APPLE_ITUNES", "DEEZER", "TWITCH", "YOUTUBE", "NETFLIX"]
    conference_protocols = ["RTMP", "TEAMSPEAK", "DISCORD", "MEET", "TEAMS", "SKYPE", 
                           "GOOGLEMEET", "SKYPECALL", "WHATSAPPCALL", "GOOGLECALL"]
    file_transfer_protocols = ["FTP_DATA", "HTTP_DOWNLOAD", "GOOGLECLOUD", "WHATSAPPFILE", 
                              "DOWNLOAD", "APPLEICLOUD", "GOOGLEDRIVE"]
    general_protocols = ["HTTP", "GMAIL", "WIKIPEDIA", "DNS", "EBAY", "TEAMVIEWER", 
                        "EASYTAXI", "MQTT", "SSH", "NTP", "ORACLE", "CITRIX", "MSSQL", "SNMP", "TWITTER", "INSTAGRAM", "FACEBOOK"]
    
    if protocol_name in streaming_protocols:
        return "STREAMING"
    elif protocol_name in conference_protocols:
        return "CONFERENCE"
    elif protocol_name in file_transfer_protocols:
        return "FILE_TRANSFER"
    elif protocol_name in general_protocols:
        return "GENERAL"
    return "OTHER"

def balance_classes_extreme(dataframe, target_samples=350000):
    """
    Balanceia as classes usando oversampling e undersampling com ruído controlado.
    
    Para amostras duplicadas, adiciona ruído proporcional usando:
    - Coeficiente de Variação (CV = std/mean)
    - Ruído multiplicativo ao invés de aditivo (evita valores negativos)
    - Clipping para manter valores dentro do intervalo original
    """
    print(f"Balanceando classes para {target_samples:,} amostras cada...")
    
    balanced_dataframes = []
    
    for class_name in dataframe['ProtocolName'].unique():
        class_data = dataframe[dataframe['ProtocolName'] == class_name].copy()
        current_count = len(class_data)
        
        if current_count > target_samples:
            sampled_data = class_data.sample(n=target_samples, random_state=42)
            sampled_data['_is_original'] = True
        else:
            sampled_data = class_data.sample(n=target_samples, replace=True, random_state=42)
            sampled_data['_is_original'] = False
            original_indices = class_data.index
            sampled_data.loc[sampled_data.index.isin(original_indices), '_is_original'] = True
            
            duplicated_mask = ~sampled_data['_is_original']
            if duplicated_mask.sum() > 0:
                numeric_columns = [col for col in sampled_data.select_dtypes(include=[np.number]).columns 
                                 if col not in ['Protocol', '_is_original']]

                for col in numeric_columns:
                    column_mean = class_data[col].mean()
                    column_std = class_data[col].std()
                    column_min = class_data[col].min()
                    column_max = class_data[col].max()
                    
                    if column_std > 0 and column_mean != 0:
                        # Calcula o coeficiente de variação
                        cv = column_std / abs(column_mean)
                        
                        # Adiciona ruído proporcional ao valor de cada amostra
                        # Isso evita valores negativos pois o ruído é relativo ao valor
                        duplicated_values = sampled_data.loc[duplicated_mask, col].values
                        
                        # Ruído proporcional: cada valor recebe ruído baseado em si mesmo
                        proportional_noise = np.random.normal(0, cv, len(duplicated_values))
                        noisy_values = duplicated_values * (1 + proportional_noise)
                        
                        # Garante que os valores fiquem dentro do intervalo original
                        # Isso evita outliers irreais e valores negativos
                        noisy_values = np.clip(noisy_values, column_min, column_max)
                        
                        sampled_data.loc[duplicated_mask, col] = noisy_values
        
        sampled_data = sampled_data.drop('_is_original', axis=1)
        balanced_dataframes.append(sampled_data)
        print(f"  {class_name}: {current_count:,} → {target_samples:,}")
    
    balanced_data = pd.concat(balanced_dataframes, ignore_index=True)
    print(f"Dataset balanceado: {len(balanced_data):,} registros")
    
    return balanced_data

def create_balanced_dataset():
    """Cria um dataset balanceado a partir do arquivo processado"""
    print("Criando dataset balanceado...")
    
    required_columns = [
        "Protocol", "Total.Fwd.Packets", "Total.Backward.Packets",
        "Total.Length.of.Bwd.Packets", "Total.Length.of.Fwd.Packets",
        "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean", "Average.Packet.Size",
        "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s", "Bwd.bytes.packet",
        "Fwd.bytes.packet", "Bwd.bytes.s", "Fwd.bytes.s", "Packets.s",
        "Speed.packet", "Fwd.Speed.packet", "Bwd.Speed.packet",
        "Ratio.Bytes", "Ratio.Packets", "ProtocolName"
    ]
    
    print("Carregando dados...")
    chunk_size = 100000
    processed_chunks = []
    
    chunks = pd.read_csv("2-processando_dataset/dataset_tratado.csv", chunksize=chunk_size)
    for i, chunk in enumerate(chunks):
        chunk = chunk[required_columns]
        processed_chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  {i+1} chunks carregados...")
    
    final_dataframe = pd.concat(processed_chunks, ignore_index=True)
    print(f"Total carregado: {len(final_dataframe):,} registros")
    
    original_size = len(final_dataframe)
    final_dataframe = final_dataframe[final_dataframe["Protocol"] != 0]
    final_dataframe["ProtocolName"] = final_dataframe["ProtocolName"].apply(categorize_protocol)
    final_dataframe = final_dataframe[final_dataframe["ProtocolName"] != "OTHER"]
    
    print(f"Após limpeza: {len(final_dataframe):,} registros")
    
    balanced_dataframe = balance_classes_extreme(final_dataframe)
    
    balanced_dataframe = balanced_dataframe.replace([np.inf, -np.inf], [1e10, -1e10])
    for col in balanced_dataframe.select_dtypes(include=[np.number]).columns:
        if col != 'ProtocolName' and balanced_dataframe[col].isnull().sum() > 0:
            balanced_dataframe[col] = balanced_dataframe[col].fillna(balanced_dataframe[col].median())
    
    output_filename = "3-balanceado_dataset/dataset_tratado_balanceado.csv"
    balanced_dataframe.to_csv(output_filename, index=False, float_format='%.8f')
    
    print(f"Dataset salvo: {output_filename}")
    print(f"Total: {len(balanced_dataframe):,} registros, {balanced_dataframe['ProtocolName'].nunique()} classes")
    
    return balanced_dataframe

if __name__ == "__main__":
    dataset = create_balanced_dataset()
    print(f"Finalizado: {len(dataset):,} registros criados.")
