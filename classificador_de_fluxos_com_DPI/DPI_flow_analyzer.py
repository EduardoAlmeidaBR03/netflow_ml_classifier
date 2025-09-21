#!/usr/bin/env python3
"""
Processador DPI para classificação de fluxos de rede.
Extrai IPs e portas de arquivos DPI e filtra fluxos por aplicação.
"""

import pandas as pd
import re
import sys
from collections import defaultdict
from pathlib import Path

class ProcessadorDPI:
    def __init__(self, applications):
        self.applications = applications if isinstance(applications, list) else [applications]

        self.dir_dpi_txt = Path('classificador_de_fluxos_com_DPI/dpi_txt')
        self.dir_flows = Path('classificador_de_fluxos_com_DPI/flows')
        self.dir_criterios = Path('classificador_de_fluxos_com_DPI/porta_ip')
        self.dir_output = Path('classificador_de_fluxos_com_DPI/flows_filtrados')

        self.dir_criterios.mkdir(exist_ok=True)
        self.dir_output.mkdir(exist_ok=True)
        
        self.regex_patterns = [
            re.compile(r'\d+\s+(TCP|UDP)\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+<->\s+(\d+\.\d+\.\d+\.\d+):(\d+)'),
            re.compile(r'(TCP|UDP)\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+<->\s+(\d+\.\d+\.\d+\.\d+):(\d+)'),
            re.compile(r'(TCP|UDP)\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+->\s+(\d+\.\d+\.\d+\.\d+):(\d+)')
        ]
        
        self.colunas_selecionadas = [
            'protocol', 'flow_duration',
            'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_bwd_pkts',
            'totlen_fwd_pkts', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean', 
            'pkt_size_avg', 'fwd_pkts_s', 'bwd_pkts_s', 'flow_byts_s', 
            'ProtocolName'
        ]
        
    def _extrair_dados_linha(self, linha):
        """Extrai dados de uma linha usando regex pré-compilados"""
        for pattern in self.regex_patterns:
            match = pattern.search(linha)
            if match:
                groups = match.groups()
                protocolo_net, ip_origem, porta_origem, ip_destino, porta_destino = groups
                
                return {
                    'IP Origem': ip_origem,
                    'Porta Origem': porta_origem,
                    'IP Destino': ip_destino,
                    'Porta Destino': porta_destino
                }
        return None
    
    def _processar_arquivo_txt(self, arquivo_path, applications_set):
        """Processa um único arquivo TXT e retorna dados para todas as aplicações"""
        dados_por_aplicacao = defaultdict(list)
        
        try:
            with open(arquivo_path, 'r', encoding='utf-8', errors='ignore') as file:
                for linha in file:
                    linha = linha.strip()
                    if not linha:
                        continue
                    
                    applications_encontradas = [app for app in applications_set if app in linha and not linha.startswith(app)]
                    
                    if applications_encontradas and ('<->' in linha or '->' in linha):
                        dados_extraidos = self._extrair_dados_linha(linha)
                        if dados_extraidos:
                            for application in applications_encontradas:
                                dados_por_aplicacao[application].append(dados_extraidos)
        
        except Exception as e:
            print(f"Erro ao processar arquivo {arquivo_path}: {e}")
        
        return dados_por_aplicacao
    
    def extrair_ips_portas(self):
        """Extrai IPs e portas dos arquivos TXT para todas as aplicações"""
        print(f"Extraindo IPs e portas das aplicações: {', '.join(self.applications)}")
        
        if not self.dir_dpi_txt.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.dir_dpi_txt}")
        
        arquivos_txt = list(self.dir_dpi_txt.glob('*.txt'))
        
        if not arquivos_txt:
            print("Nenhum arquivo .txt encontrado")
            return False
        
        applications_set = set(self.applications)
        total_processados = 0
        
        for arquivo_txt in arquivos_txt:
            nome_arquivo = arquivo_txt.stem
            dados_por_aplicacao = self._processar_arquivo_txt(arquivo_txt, applications_set)
            
            for application, dados in dados_por_aplicacao.items():
                if dados:
                    df = pd.DataFrame(dados)
                    arquivo_criterios = self.dir_criterios / f'{nome_arquivo}{application.lower()}.csv'
                    df.to_csv(arquivo_criterios, index=False)
                    total_processados += 1
        
        print(f"✓ Extração concluída: {total_processados} arquivos processados")
        return total_processados > 0
    
    def _carregar_e_preparar_dados(self, arquivo_flows, arquivo_criterios):
        """Carrega e prepara dados de fluxos e critérios"""
        try:
            df_fluxos = pd.read_csv(arquivo_flows)
            df_criterios = pd.read_csv(arquivo_criterios)
            
            df_fluxos = df_fluxos.dropna(subset=['src_port', 'dst_port', 'src_ip', 'dst_ip'])
            df_criterios = df_criterios.dropna(subset=['Porta Origem', 'Porta Destino', 'IP Origem', 'IP Destino'])
            
            for col in ['src_port', 'dst_port']:
                df_fluxos[col] = pd.to_numeric(df_fluxos[col], errors='coerce')
            
            for col in ['Porta Origem', 'Porta Destino']:
                df_criterios[col] = pd.to_numeric(df_criterios[col], errors='coerce')
            
            df_fluxos = df_fluxos.dropna(subset=['src_port', 'dst_port'])
            df_criterios = df_criterios.dropna(subset=['Porta Origem', 'Porta Destino'])
            
            return df_fluxos, df_criterios
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return None, None
    
    def filtrar_fluxos(self):
        """Filtra os fluxos usando os IPs e portas extraídos para todas as aplicações"""
        print("Filtrando fluxos de rede...")
        
        total_processados = 0
        
        for application_original in self.applications:
            application_lower = application_original.lower()
            application_upper = application_original.upper()
            
            arquivos_criterios = list(self.dir_criterios.glob(f'*{application_lower}.csv'))
            
            for arquivo_criterios in arquivos_criterios:
                nome_base = arquivo_criterios.name.replace(f'{application_lower}.csv', '')
                arquivo_flows = self.dir_flows / f'{nome_base}.csv'
                
                if not arquivo_flows.exists():
                    continue
                
                df_fluxos, df_criterios = self._carregar_e_preparar_dados(arquivo_flows, arquivo_criterios)
                
                if df_fluxos is None or df_criterios is None or len(df_fluxos) == 0 or len(df_criterios) == 0:
                    continue
                
                ips_origem_set = set(df_criterios['IP Origem'])
                ips_destino_set = set(df_criterios['IP Destino'])
                portas_origem_set = set(df_criterios['Porta Origem'])
                portas_destino_set = set(df_criterios['Porta Destino'])
                
                mascara = (
                    df_fluxos['src_ip'].isin(ips_origem_set) &
                    df_fluxos['dst_ip'].isin(ips_destino_set) &
                    df_fluxos['src_port'].isin(portas_origem_set) &
                    df_fluxos['dst_port'].isin(portas_destino_set)
                )
                
                fluxos_filtrados = df_fluxos[mascara].copy()
                
                if len(fluxos_filtrados) == 0:
                    continue
                
                fluxos_filtrados['ProtocolName'] = application_upper
                
                colunas_existentes = [col for col in self.colunas_selecionadas if col in fluxos_filtrados.columns]
                fluxos_filtrados_final = fluxos_filtrados[colunas_existentes]
                
                output_file = self.dir_output / f'{nome_base}{application_upper}.csv'
                fluxos_filtrados_final.to_csv(output_file, index=False)
                
                total_processados += 1
        
        print(f"✓ Filtragem concluída: {total_processados} arquivos gerados")
        return total_processados > 0
    
    def processar(self):
        """Executa o processo completo: extração de IPs/portas seguida da filtragem de fluxos"""
        print(f"🚀 Processando aplicações: {', '.join(self.applications)}")
        
        try:
            if not self.extrair_ips_portas():
                print("❌ Falha na extração de IPs e portas")
                return False
            
            if not self.filtrar_fluxos():
                print("❌ Falha na filtragem de fluxos")
                return False
            
            print("🎉 Processamento concluído com sucesso!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Arquivo/diretório não encontrado: {e}")
            return False
        except PermissionError as e:
            print(f"❌ Erro de permissão: {e}")
            return False
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            return False

def main():
    """Função principal que processa argumentos da linha de comando"""
    application_default = "skype"
    
    if len(sys.argv) == 1:
        applications = [application_default]
    elif len(sys.argv) == 2:
        applications = [sys.argv[1]]
    else:
        applications = sys.argv[1:]
    
    if len(applications) == 0:
        print("Uso: python DPI_flow_analyzer.py [aplicacao1] [aplicacao2] [...]")
        sys.exit(1)
    
    processador = ProcessadorDPI(applications)
    sucesso = processador.processar()
    
    sys.exit(0 if sucesso else 1)

if __name__ == "__main__":
    main()
