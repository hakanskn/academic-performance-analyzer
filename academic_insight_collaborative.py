import argparse
import sys

import openai
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anthropic import Anthropic
from fpdf import FPDF
import numpy as np
from datetime import datetime
import os

from openai import OpenAI

from config_manager import ConfigManager
from cost_calculator import CostCalculator
import os
import logging
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from contextlib import contextmanager


# academic_insight_collaborative.py dosyasının en başına ekleyin
import os

import locale

# Varsayılan kodlamayı UTF-8 olarak ayarla
# sys.stdout.reconfigure(encoding='utf-8')
# sys.stderr.reconfigure(encoding='utf-8')
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class ImprovedAcademicAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.logger = self._setup_logging()
        self.temp_files: List[Path] = []
        self.student_info = {}
        self.subject_data = {}  # Tüm ders verilerini tutacak dictionary

        # API istemcilerini oluştur
        config = ConfigManager()
        api_keys = config.get_api_keys()

        self.openai_client = OpenAI(api_key=api_keys.get("openai"))
        self.anthropic_client = Anthropic(api_key=api_keys.get("anthropic"))

        self._setup_environment()

    def _setup_logging(self) -> logging.Logger:
        """Loglama sistemi kurulumu"""
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler('analysis.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_environment(self):
        """Çalışma ortamı hazırlığı"""
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    @contextmanager
    def _pdf_handler(self):
        """PDF dosyası için güvenli context manager"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                yield pdf
        except Exception as e:
            self.logger.error(f"PDF okuma hatası: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_api_request(self, endpoint: str, data: Dict) -> Dict:
        """API istekleri için yeniden deneme mekanizmalı method"""
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=data) as response:
                response.raise_for_status()
                return await response.json()

    # @lru_cache(maxsize=128)
    # def _process_subject_data(self, subject: str) -> pd.DataFrame:
    #     """Ders verisi işleme (cache'li)"""
    #     try:
    #         with self._pdf_handler() as pdf:
    #             data = self._extract_subject_data(pdf, subject)
    #             return self._validate_and_clean_data(data)
    #     except Exception as e:
    #         self.logger.error(f"{subject} dersi işlenirken hata: {str(e)}")
    #         raise

    # def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Veri temizleme ve doğrulama"""
    #     # Sayısal değerler için validasyon
    #     numeric_cols = ['Başarı Oranı %', 'Soru Sayısı']
    #     for col in numeric_cols:
    #         if col in df.columns:
    #             # Geçersiz değerleri NaN'a çevir
    #             df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    #
    #             # Aralık kontrolü
    #             if col == 'Başarı Oranı %':
    #                 invalid_mask = (df[col] < 0) | (df[col] > 100)
    #                 if invalid_mask.any():
    #                     self.logger.warning(f"Geçersiz başarı oranları tespit edildi")
    #                     df.loc[invalid_mask, col] = None
    #
    #     return df

    async def analyze_performance(self):
        """Performans analizi (asenkron)"""
        try:
            analysis_results = {}

            # subject_data içindeki dersleri kullan
            for subject, original_df in self.subject_data.items():
                # DataFrame'in tam bir kopyasını oluştur
                df = original_df.copy()

                # Sütun isimlerini belirle
                df.columns = ["Sıra", "Kazanım Adı", "1 Soru Sayısı", "1 Başarı Oranı",
                              "2 Soru Sayısı", "2 Başarı Oranı", "3 Soru Sayısı", "3 Başarı Oranı",
                              "4 Soru Sayısı", "4 Başarı Oranı", "5 Soru Sayısı", "5 Başarı Oranı",
                              "6 Soru Sayısı", "6 Başarı Oranı", "7 Soru Sayısı", "7 Başarı Oranı",
                              "Toplam Soru Sayısı", "Toplam Başarı Oranı"]

                # İlk iki satırı atla ve yeni bir DataFrame oluştur
                df = df.iloc[2:].reset_index(drop=True)

                # Başarı oranlarını dönüştür
                columns_to_convert = ['1 Başarı Oranı', '2 Başarı Oranı', '3 Başarı Oranı',
                                      '4 Başarı Oranı', '5 Başarı Oranı', '6 Başarı Oranı',
                                      '7 Başarı Oranı', 'Toplam Başarı Oranı']

                # Tüm başarı oranlarını bir kerede dönüştür
                for column in columns_to_convert:
                    df[column] = pd.to_numeric(df[column].str.replace(',', '.'), errors='coerce')

                # Analizleri hesapla
                total_success = df['Toplam Başarı Oranı']
                analysis = {
                    'mean_success': total_success.mean(),
                    'strong_topics': df[total_success >= 75]['Kazanım Adı'].tolist(),
                    'developing_topics': df[(total_success >= 50) &
                                            (total_success < 75)]['Kazanım Adı'].tolist(),
                    'priority_topics': df[total_success < 50]['Kazanım Adı'].tolist()
                }

                # İstatistiksel analizler
                analysis['statistics'] = {
                    'min_success': total_success.min(),
                    'max_success': total_success.max(),
                    'median_success': total_success.median(),
                    'total_topics': len(df)
                }

                analysis_results[subject] = analysis

            self.logger.info("Performans analizi tamamlandı")
            # print(f"\nAnaliz sonuçları: {analysis_results}")  # Debug için eklendi
            return analysis_results

        except Exception as e:
            self.logger.error(f"Performans analizi hatası: {str(e)}")
            raise

    async def create_visualizations(self, subject_data):
        """Görsel analizlerin oluşturulması"""
        # Debug için mevcut verileri yazdır
        # print("\n=== Görselleştirme Debug ===")
        # print("1. Gelen subject_data:")
        # for subject, df in subject_data.items():
            # print(f"\n{subject} DataFrame:")
            # print(f"Sütunlar: {df.columns.tolist()}")
            # print(f"Satır sayısı: {len(df)}")
            # print(f"İlk 3 satır:\n{df.head(3)}")

        try:
            visualizations = {}
            viz_dir = self.temp_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            for subject, original_df in subject_data.items():
                # DataFrame'i temizle ve hazırla
                df = original_df.copy()

                # None sütunlarını kaldır
                # df = df.loc[:, df.columns.notna()]

                # İlk iki satırı atla
                # df = df.iloc[2:].reset_index(drop=True)

                # İlk iki ve son iki sütunu seçip, geri kalanları silmek
                df = pd.concat([df.iloc[:, :2], df.iloc[:, -2:]], axis=1)

                # Sütun isimlerini temizle ve yeniden adlandır
                # new_columns = {
                #     df.columns[0]: "Sıra",
                #     df.columns[1]: "Kazanım Adı",
                #     df.columns[-2]: "Toplam Soru Sayısı",
                #     df.columns[-1]: "Toplam Başarı Oranı"
                # }

                df.columns = [["Sıra", "Kazanım Adı", "Toplam Soru Sayısı", "Toplam Başarı Oranı"]]

                # df = df.rename(columns=new_columns)

                self.logger.info(f"Temizlenmiş sütunlar: {df.columns.tolist()}")
                subject_viz = {}

                # 1. Başarı dağılımı pasta grafiği
                plt.figure(figsize=(10, 6))

                # Başarı oranlarını düzenle
                success_rates = pd.to_numeric(
                    df['Toplam Başarı Oranı'].squeeze().astype(str).str.replace(',', '.'),
                    errors='coerce'
                ).dropna()

                if not success_rates.empty:
                    categories = {
                        'Mükemmel (>90%)': len(success_rates[success_rates >= 90]),
                        'İyi (75-90%)': len(success_rates[(success_rates >= 75) & (success_rates < 90)]),
                        'Gelişmekte (50-75%)': len(success_rates[(success_rates >= 50) & (success_rates < 75)]),
                        'Geliştirilmeli (<50%)': len(success_rates[success_rates < 50])
                    }

                    if sum(categories.values()) > 0:
                        plt.pie(categories.values(),
                                labels=categories.keys(),
                                autopct='%1.1f%%',
                                colors=['#2ecc71', '#3498db', '#f1c40f', '#e74c3c'])
                        plt.title(f'{subject} - Başarı Dağılımı')

                        pie_path = viz_dir / f'{subject}_distribution.png'
                        plt.savefig(pie_path)
                        plt.close()
                        subject_viz['distribution'] = pie_path

                        # 2. Konu bazlı başarı oranları
                        plt.figure(figsize=(12, 6))

                        # Başarı oranlarına göre sırala
                        topic_success = pd.DataFrame({
                            'topic': df['Kazanım Adı'].squeeze(),
                            'success_rate': success_rates.squeeze()
                        })

                        # En iyi ve en kötü 5'i al
                        top_5 = topic_success.nlargest(5, 'success_rate')
                        bottom_5 = topic_success.nsmallest(5, 'success_rate')
                        top_bottom = pd.concat([top_5, bottom_5])

                        # Renkleri ayarla: En iyi konular için yeşil, en kötü konular için kırmızı
                        colors = ['#2ecc71'] * len(top_5) + ['#e74c3c'] * len(bottom_5)

                        plt.barh(range(len(top_bottom)), top_bottom['success_rate'], color=colors)
                        plt.yticks(range(len(top_bottom)), top_bottom['topic'].str[:50])
                        plt.xlabel('Başarı Oranı (%)')
                        plt.title(f'{subject} - En İyi ve Geliştirilmesi Gereken Konular')

                        # plt.barh(range(len(top_bottom)), top_bottom['success_rate'])
                        # plt.yticks(range(len(top_bottom)), top_bottom['topic'].str[:50])
                        # plt.xlabel('Başarı Oranı (%)')
                        # plt.title(f'{subject} - En İyi ve Geliştirilmesi Gereken Konular')

                        topics_path = viz_dir / f'{subject}_topics.png'
                        plt.savefig(topics_path, bbox_inches='tight')
                        plt.close()
                        subject_viz['topics'] = topics_path

                visualizations[subject] = subject_viz

            # Genel karşılaştırma grafiği
            if visualizations:
                plt.figure(figsize=(10, 6))
                means = {}

                for subj, viz_data in visualizations.items():
                    if 'distribution' in viz_data:
                        df_copy = subject_data[subj].copy()

                        df_copy = pd.concat([df_copy.iloc[:, :2], df_copy.iloc[:, -2:]], axis=1)

                        df_copy.columns = [["Sıra", "Kazanım Adı", "Toplam Soru Sayısı", "Toplam Başarı Oranı"]]

                        success_rates = pd.to_numeric(
                            df_copy['Toplam Başarı Oranı'].squeeze().astype(str).str.replace(',', '.'),
                            errors='coerce'
                        )
                        means[subj] = success_rates.mean()

                if means:
                    plt.bar(means.keys(), means.values())
                    plt.ylabel('Ortalama Başarı (%)')
                    plt.title('Dersler Arası Karşılaştırma')
                    plt.xticks(rotation=45)

                    comparison_path = viz_dir / 'subject_comparison.png'
                    plt.savefig(comparison_path, bbox_inches='tight')
                    plt.close()
                    visualizations['overall'] = {'comparison': comparison_path}

            self.logger.info("Görsel analizler tamamlandı")
            return visualizations

        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {str(e)}")
            raise

    async def conduct_ai_discussion(self, analysis_results):
        """AI ajanları arasında tartışma yürütme"""
        try:
            # Sistem promptlarını tanımla

            system_prompts = {
                "data_expert": """Sen sınav performans verilerini analiz eden, öğrencilere koçluk yapan bir eğitim danışmanısın.

                YAKLAŞIM:
                • Samimi ama profesyonel bir dil kullan
                • 8. sınıf seviyesine uygun şekilde LGS odaklı değerlendirme yap
                • Her eleştiriyi bir fırsat olarak sun
                • Gençlerin dilinden anla ve onları motive et

                ANALİZ ODAĞI:
                • Kazanımlar arası bağlantıları vurgula
                • LGS'de çıkması muhtemel soru tiplerini düşün
                • Güçlü yönleri öne çıkar
                • Gelişim alanlarını "henüz" diyerek ifade et

                DEĞERLENDİRME YAKLAŞIMI:
                • "Harika ilerleme görüyorum!"
                • "Bu konuda gerçekten iyisin, diğer konularda da aynı başarıyı yakalayabilirsin"
                • "Şu an için zorlanıyor olabilirsin ama birlikte üstesinden geleceğiz"
                • "LGS'ye kadar bu konuyu rahatlıkla halledebiliriz"

                MOTİVASYON ODAĞI:
                • Her zaman yapıcı ol
                • Başarılabilir hedefler koy
                • İlerlemeyi somut örneklerle göster
                • Olumlu yönleri mutlaka vurgula""",

                "education_expert": """Sen bir eğitim koçusun. 8. sınıf öğrencilerine LGS hazırlık sürecinde destek oluyorsun.

                TEMEL YAKLAŞIM:
                • Arkadaşça ve destekleyici ol
                • Sınav kaygısını azaltıcı bir dil kullan
                • Çalışma motivasyonunu artır
                • Her konuyu LGS perspektifinden değerlendir

                ÖĞRENME STRATEJİLERİ:
                • Her konu için özel çalışma teknikleri öner
                • Zaman yönetimi için ipuçları ver
                • Soru çözme stratejileri geliştir
                • Konular arası bağlantılar kur

                DÖNÜT VERME:
                • "Bu konuda süper gidiyorsun!"
                • "Bak burada çok güzel bir gelişme var"
                • "Şu stratejiyi denersen daha da iyi olacak"
                • "Bu konuyu hallettiğinde diğerleri çok daha kolay gelecek"

                MOTİVASYON VE DESTEK:
                • Her küçük ilerlemeyi takdir et
                • Zorlukları normal göster
                • Başarı hikayelerinden örnekler ver
                • Her zaman "yapabilirsin" mesajı ver"""
            }

            # system_prompts = {
            #     "data_expert": """Sen bir eğitsel veri analisti uzmanısın. Öğrencinin sınav performans verilerini analiz edeceksin.
            #
            #     GÖREVLER:
            #     1. Kazanım Analizi:
            #        - Başarı oranlarını sınıflandır (Yüksek: >75%, Orta: 50-75%, Düşük: <50%)
            #        - Her kategorideki kazanım sayılarını belirt
            #        - Kritik kazanımları önceliklendir
            #
            #     2. Trend Analizi:
            #        - Benzer kazanımlardaki performans paternlerini belirle
            #        - Kazanımlar arası başarı farklarını açıkla
            #        - Düşük performanslı alanların olası sebeplerini analiz et
            #
            #     3. Gelişim Fırsatları:
            #        - Hangi kazanımların gelişimi diğerlerini olumlu etkileyecek?
            #        - Hangi konular arasında güçlü bağlantılar var?
            #        - Optimal gelişim sıralaması nasıl olmalı?
            #
            #     4. Veri Odaklı Öneriler:
            #        - Her kazanım kategorisi için spesifik stratejiler sun
            #        - Başarılı olduğu konulardaki yöntemleri diğerlerine nasıl uyarlayabiliriz?
            #        - Çalışma süresini nasıl optimize edebiliriz?
            #
            #     ÇIKTI FORMATI:
            #     1. Sayısal Değerlendirme: [İstatistiksel analiz ve trendler]
            #     2. Kritik Noktalar: [Öncelikli gelişim alanları]
            #     3. Başarı Deseni: [Başarılı ve başarısız konular arasındaki ilişkiler]
            #     4. Veri Odaklı Öneriler: [Somut, ölçülebilir öneriler]""",
            #
            #     "education_expert": """Sen bir eğitim ve öğrenme uzmanısın. Öğrencinin öğrenme sürecini analiz edip iyileştirme önerileri sunacaksın.
            #
            #     GÖREVLER:
            #     1. Öğrenme Analizi:
            #        - Güçlü ve zayıf olduğu öğrenme alanlarını belirle
            #        - Kazanımlar arasındaki öğrenme bağlantılarını tespit et
            #        - Olası öğrenme engellerini tanımla
            #
            #     2. Pedagojik Değerlendirme:
            #        - Her kazanım türü için uygun öğrenme stratejileri belirle
            #        - Eksik ön öğrenmeleri tespit et
            #        - Kavram yanılgılarını belirle
            #
            #     3. Kişiselleştirilmiş Plan:
            #        - Öğrencinin başarılı olduğu yöntemleri tespit et
            #        - Zayıf alanlarda alternatif öğrenme yaklaşımları öner
            #        - Bireysel öğrenme hızını dikkate al
            #
            #     4. Motivasyon Stratejileri:
            #        - Başarı hikayelerini kullan
            #        - Kısa ve uzun vadeli hedefler belirle
            #        - İlerlemeyi görünür kıl
            #
            #     ÇIKTI FORMATI:
            #     1. Öğrenme Profili: [Güçlü ve gelişim alanları]
            #     2. Engel Analizi: [Öğrenme önündeki engeller ve çözümler]
            #     3. Gelişim Planı: [Kişiselleştirilmiş öğrenme stratejileri]
            #     4. Motivasyon Önerileri: [Somut motivasyon artırıcı taktikler]"""
            # }

            print("\n=== AI TARTIŞMASI BAŞLIYOR ===")
            discussion_results = {}

            for subject, analysis in analysis_results.items():
                print(f"\n{subject} Dersi Değerlendirmesi:")
                self.logger.info(f"{subject} dersi için AI tartışması başlatılıyor...")

                # Veri analisti için detaylı prompt hazırla
                data_prompt = f"""
                {subject} dersi analiz sonuçları:
                - Ortalama başarı: {analysis['mean_success']:.1f}%
                - Güçlü konular ({len(analysis['strong_topics'])} adet): {', '.join(analysis['strong_topics'][:3])}...
                - Geliştirilmesi gereken konular ({len(analysis['priority_topics'])} adet): {', '.join(analysis['priority_topics'][:3])}...

                İstatistikler:
                - En yüksek başarı: {analysis['statistics']['max_success']:.1f}%
                - En düşük başarı: {analysis['statistics']['min_success']:.1f}%
                - Medyan başarı: {analysis['statistics']['median_success']:.1f}%
                - Toplam kazanım: {analysis['statistics']['total_topics']}

                Lütfen verilen system prompt doğrultusunda detaylı bir analiz sun.
                """

                try:
                    print("\nVeri Analisti Konuşuyor:")
                    data_response = self._make_api_request_openai(
                        client=self.openai_client,
                        system_prompt=system_prompts["data_expert"],
                        user_prompt=data_prompt
                    )
                    print(data_response)

                except Exception as e:
                    self.logger.error(f"GPT-4 API hatası: {str(e)}")
                    data_response = "Veri analizi yapılamadı."

                # Eğitim uzmanı için veri analistinin yorumlarını da içeren prompt hazırla
                edu_prompt = f"""
                {subject} dersi için yapılan analiz ve veri uzmanının değerlendirmesi:

                ANALİZ SONUÇLARI:
                - Ortalama başarı: {analysis['mean_success']:.1f}%
                - Güçlü konular: {', '.join(analysis['strong_topics'][:3])}...
                - Geliştirilmesi gereken konular: {', '.join(analysis['priority_topics'][:3])}...

                VERİ UZMANI DEĞERLENDİRMESİ:
                {data_response}

                Lütfen verilen system prompt doğrultusunda pedagojik bir değerlendirme yap ve
                veri uzmanının tespitlerini de dikkate alarak öğrenme stratejileri öner.
                """

                try:
                    print("\nEğitim Uzmanı Konuşuyor:")
                    edu_response = self._make_api_request_anthropic(
                        client=self.anthropic_client,
                        system_prompt=system_prompts["education_expert"],
                        user_prompt=edu_prompt
                    )
                    print(edu_response)

                except Exception as e:
                    self.logger.error(f"Claude API hatası: {str(e)}")
                    edu_response = "Eğitsel analiz yapılamadı."

                # Sonuçları yapılandırılmış formatta birleştir
                discussion_results[subject] = {
                    'data_analysis': {
                        'raw_response': data_response,
                        'structured_analysis': self._parse_data_analysis(data_response)
                    },
                    'education_analysis': {
                        'raw_response': edu_response,
                        'structured_analysis': self._parse_education_analysis(edu_response)
                    },
                    'combined_recommendations': self._combine_recommendations(
                        data_response, edu_response
                    )
                }

            self.logger.info("AI tartışması tamamlandı")
            return discussion_results

        except Exception as e:
            self.logger.error(f"AI tartışması hatası: {str(e)}")
            raise

    def _parse_data_analysis(self, response):
        """Veri analisti yanıtını yapılandırılmış formata dönüştür"""
        sections = {
            'numerical_evaluation': [],
            'critical_points': [],
            'success_pattern': [],
            'data_driven_suggestions': []
        }

        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if 'Sayısal Değerlendirme:' in line:
                current_section = 'numerical_evaluation'
            elif 'Kritik Noktalar:' in line:
                current_section = 'critical_points'
            elif 'Başarı Deseni:' in line:
                current_section = 'success_pattern'
            elif 'Veri Odaklı Öneriler:' in line:
                current_section = 'data_driven_suggestions'
            elif line and current_section:
                sections[current_section].append(line)

        return sections

    def _parse_education_analysis(self, response):
        """Eğitim uzmanı yanıtını yapılandırılmış formata dönüştür"""
        sections = {
            'learning_profile': [],
            'obstacle_analysis': [],
            'development_plan': [],
            'motivation_suggestions': []
        }

        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if 'Öğrenme Profili:' in line:
                current_section = 'learning_profile'
            elif 'Engel Analizi:' in line:
                current_section = 'obstacle_analysis'
            elif 'Gelişim Planı:' in line:
                current_section = 'development_plan'
            elif 'Motivasyon Önerileri:' in line:
                current_section = 'motivation_suggestions'
            elif line and current_section:
                sections[current_section].append(line)

        return sections

    def _identify_subject(self, table):
        """Tablonun hangi derse ait olduğunu belirle"""
        subject_mapping = {
            "TÜRKÇE": "Türkçe",
            "MATEMATİK": "Matematik",
            "FEN BİLİMLERİ": "Fen Bilimleri",
            "İNKILAP TARİHİ": "İnkılap Tarihi",
            "DİN KÜLTÜRÜ": "Din Kültürü",
            "YABANCI DİL": "İngilizce"
        }

    # def _format_bullet_point(self, text):
    #     """Metni noktalı madde işareti formatına dönüştür"""
    #     text = text.strip()
    #     # Varolan tire veya yıldız işaretlerini kaldır
    #     text = text.replace('- ', '').replace('* ', '')
    #     if not text.startswith('•'):
    #         text = f"• {text}"
    #     return text

    async def generate_report(self, student_info, analysis_results, visualizations, discussion_results, output_dir):
        """Final PDF raporu oluştur"""
        try:
            from fpdf import FPDF

            class UTF8Report(FPDF):
                def __init__(self):
                    super().__init__()
                    self.add_font('Arial', '', r"C:\Windows\Fonts\arial.ttf", uni=True)
                    self.add_font('Arial', 'B', r"C:\Windows\Fonts\arialbd.ttf", uni=True)
                    self.add_header_footer = True

                def header(self):
                    if self.add_header_footer:  # Eğer header aktifse, işle
                        self.set_font('Arial', 'B', 9)
                        self.cell(80)
                        self.cell(30, 10, 'Akademik Performans Analiz Raporu', 0, 0, 'C')
                        self.ln(20)

                def footer(self):
                    if self.add_header_footer:  # Eğer header aktifse, işle
                        self.set_y(-15)
                        self.set_font('Arial', '', 8)
                        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

                def add_page(self, *args, **kwargs):
                    # İlk sayfada header/footer'ı devre dışı bırak
                    if len(self.pages) == 0:  # İlk sayfa kontrolü
                        self.add_header_footer = False
                    else:
                        self.add_header_footer = True
                    super().add_page(*args, **kwargs)

                def chapter_title(self, title):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, title, 0, 1, 'L')
                    self.ln(4)

                def chapter_body(self, text):
                    """PDF'de metin içeriğini formatlar ve yerleştirir."""
                    self.set_font('Arial', '', 10)

                    # Sayfa boyutları ve kenar boşlukları
                    effective_page_width = self.w - (self.l_margin + self.r_margin + 20)  # Kullanılabilir genişlik
                    line_height = 6  # Satır yüksekliği
                    paragraph_spacing = 3  # Paragraflar arası boşluk
                    indent_width = 10  # Madde imi girintisi

                    def calc_text_width(text):
                        """Metnin genişliğini hesaplar"""
                        return self.get_string_width(text)

                    def format_line(text, max_width):
                        """Metni belirli genişliğe göre biçimlendirir"""
                        words = text.split()
                        lines = []
                        current_line = []
                        current_width = 0

                        for word in words:
                            word_width = calc_text_width(word + ' ')
                            if current_width + word_width <= max_width:
                                current_line.append(word)
                                current_width += word_width
                            else:
                                if current_line:
                                    lines.append(' '.join(current_line))
                                current_line = [word]
                                current_width = word_width

                        if current_line:
                            lines.append(' '.join(current_line))
                        return lines

                    def write_bullet_point(text):
                        """Madde imli metni yazar"""
                        bullet_text = text.lstrip('•- *').strip()
                        max_text_width = effective_page_width - indent_width

                        # İlk satır için madde imi ekle
                        self.set_x(self.l_margin + indent_width)
                        self.cell(indent_width / 2, line_height, '•')

                        # Metni böl ve yaz
                        lines = format_line(bullet_text, max_text_width)
                        for i, line in enumerate(lines):
                            if i == 0:
                                self.set_x(self.l_margin + indent_width * 1.5)
                            else:
                                self.set_x(self.l_margin + indent_width * 1.5)
                            self.multi_cell(max_text_width, line_height, line)

                        self.ln(paragraph_spacing)

                    def write_normal_text(text):
                        """Normal metni yazar"""
                        lines = format_line(text, effective_page_width)
                        for line in lines:
                            if self.get_y() > self.h - 30:  # Sayfa sonu kontrolü
                                self.add_page()
                            self.multi_cell(effective_page_width, line_height, line, align='J')
                        self.ln(paragraph_spacing)

                    # Ana metin işleme
                    paragraphs = text.split('\n')
                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if not paragraph:
                            self.ln(paragraph_spacing)
                            continue

                        # Sayfa kontrolü
                        if self.get_y() > self.h - 30:
                            self.add_page()

                        # Madde imli mi normal mi kontrol et
                        if paragraph.lstrip().startswith(('•', '-', '*')):
                            write_bullet_point(paragraph)
                        else:
                            write_normal_text(paragraph)

                    self.ln(paragraph_spacing)

                # def chapter_body(self, text):
                #     self.set_font('Arial', '', 10)  # Daha küçük font boyutu
                #     import textwrap
                #     wrapped_text = "\n".join(textwrap.wrap(text, width=100))  # Uzun metni kır
                #     try:
                #         if self.get_y() > self.h - 20:  # Eğer sayfa bitiyorsa
                #             self.add_page()
                #         self.multi_cell(0, 10, wrapped_text)  # Sabit genişlikte metni ekle
                #     except Exception as e:
                #         print(f"Metni eklerken hata oluştu: {e}")

                # def chapter_body(self, text):
                #     from textwrap3 import wrap
                #     # wrapped_text = "\n".join(wrap(text, width=100))  # Uzun metni kır
                #     self.set_font('Arial', '', 11)
                #     # Eğer metin birden fazla satırsa ve madde işareti içeriyorsa her satırı ayrı formatlayalım
                #     wrapped_line = ''
                #     if '\n' in text:
                #         for line in text.split('\n'):
                #             if line.strip():  # Boş satırları atla
                #                 if '•' in line or '-' in line or '*' in line:
                #                     line = self._format_bullet_point(line)
                #
                #                 wrapped_line = "\n".join(wrap(line, width=100))
                #                 self.multi_cell(200, 10, self.safe_encode(wrapped_line))
                #     else:
                #         wrapped_text = "\n".join(wrap(text, width=100))
                #         self.multi_cell(200, 10, self.safe_encode(wrapped_text))
                #     self.ln()

                @staticmethod
                def safe_encode(text):
                    try:
                        text.encode('latin-1')
                        return text
                    except UnicodeEncodeError:
                        return text.encode('utf-8').decode('utf-8')

                @staticmethod
                def _format_bullet_point(text):
                    """Metni noktalı madde işareti formatına dönüştür"""
                    text = text.strip()
                    # Varolan tire veya yıldız işaretlerini kaldır
                    text = text.replace('- ', '').replace('* ', '')

                    # Eğer başında bullet point yoksa ekle
                    if not text.startswith('•'):
                        text = f"• {text}"
                    elif not text.startswith('• '):  # Bullet point var ama boşluk yoksa
                        text = f"• {text[1:]}"

                    # Cümle sonunda nokta yoksa ekle
                    if not text.endswith('.'):
                        text = f"{text}."

                    return text


            # PDF oluştur
            pdf = UTF8Report()
            pdf.add_page()

            # Sayfa genişliği ve yüksekliği
            page_width = pdf.w  # 210mm (A4)

            # Logo ekle (ortada, üstte)
            logo_path = "assets/yonelt_logo.png"
            logo_width = 70  # logo genişliği (mm)
            logo_x = (page_width - logo_width) / 2  # merkeze hizalama
            pdf.image(logo_path, x=logo_x, y=20, w=logo_width)

            # Logo ile başlık arası boşluk
            pdf.ln(80)  # Logo altında yeterli boşluk

            # Başlık
            pdf.set_font('Arial', 'B', 30)
            pdf.cell(0, 10, 'AKADEMİK PERFORMANS ANALİZ RAPORU', 0, 1, 'C')
            pdf.ln(20)  # Başlık ile okul arası boşluk

            # Okul adı
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f"{student_info.get('school', 'Belirtilmemiş')}", 0, 1, 'C')
            pdf.ln(10)  # Okul ile öğrenci arası boşluk

            # Öğrenci adı soyadı
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"{student_info.get('name', 'Belirtilmemiş')}", 0, 1, 'C')
            pdf.ln(10)  # Öğrenci ile tarih arası boşluk

            # Tarih
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"{student_info.get('report_date', 'Belirtilmemiş')}", 0, 1, 'C')

            # İlk sayfanın geri kalanı boş kalsın
            pdf.add_page()  # Yeni sayfadan başla

            # Her ders için analiz
            for subject, analysis in analysis_results.items():
                pdf.add_page()
                pdf.chapter_title(f"{subject} Analizi")

                # İstatistiksel bilgiler
                pdf.chapter_body(f"""
                Ortalama Basari: %{analysis['mean_success']:.1f} \n
                En Yuksek Basari: %{analysis['statistics']['max_success']:.1f} \n
                En Dusuk Basari: %{analysis['statistics']['min_success']:.1f} \n
                """)

                # Görselleştirmeler
                if subject in visualizations:
                    viz_data = visualizations[subject]
                    if 'distribution' in viz_data:
                        pdf.image(str(viz_data['distribution']), x=10, w=190)
                    if 'topics' in viz_data:
                        # pdf.add_page()
                        pdf.image(str(viz_data['topics']), x=10, w=190)

                # AI Değerlendirmeleri
                if subject in discussion_results:
                    pdf.add_page()
                    pdf.chapter_title(f"{subject} - Uzman Degerlendirmeleri")

                    # Veri Analisti değerlendirmesi
                    if 'data_analysis' in discussion_results[subject]:
                        pdf.chapter_title("Veri Analizi:")
                        # Raw response kullanıyoruz
                        data_response = discussion_results[subject]['data_analysis'].get('raw_response', '')
                        pdf.chapter_body(data_response)

                    # Eğitim Uzmanı değerlendirmesi
                    if 'education_analysis' in discussion_results[subject]:
                        pdf.chapter_title("Eğitsel Analiz:")
                        # Raw response kullanıyoruz
                        edu_response = discussion_results[subject]['education_analysis'].get('raw_response', '')
                        pdf.chapter_body(edu_response)

                    # for heading, items in data_analysis.items():
                    #     if items:
                    #         pdf.chapter_title(heading.replace('_', ' ').title())
                    #         for item in items:
                    #             # Burada madde imi formatlaması ekliyoruz
                    #             formatted_text = self._format_bullet_point(item)
                    #             pdf.chapter_body(formatted_text)
                    #             pdf.chapter_body(f"• {item}")

            # Genel karşılaştırma
            if 'overall' in visualizations and 'comparison' in visualizations['overall']:
                pdf.add_page()
                pdf.chapter_title('DERSLER ARASI KARSILASTIRMA')
                pdf.image(str(visualizations['overall']['comparison']), x=10, w=190)

            # Raporu kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = ''.join(c if c.isalnum() else '_' for c in student_info['name'])
            report_path = Path(output_dir) / f"akademik_analiz_{safe_name}_{timestamp}.pdf"
            # pdf.output(str(report_path))
            pdf.output(str(report_path).encode('utf-8').decode('utf-8'))

            self.logger.info(f"Rapor olusturuldu: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Rapor olusturma hatasi: {str(e)}")
            raise

    def _make_api_request_openai(self, client, system_prompt, user_prompt):
        """OpenAI API isteği gönder (senkron)"""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API hatası: {str(e)}")
            raise

    def _make_api_request_anthropic(self, client, system_prompt, user_prompt):
        """Anthropic API isteği gönder (senkron)"""
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "assistant", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Anthropic API hatası: {str(e)}")
            raise

    def _combine_recommendations(self, data_response, edu_response):
        """İki uzmanın önerilerini birleştir"""
        combined = []

        # Basit metin analizi ile önerileri çıkar
        for response in [data_response, edu_response]:
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['öner', 'tavsiye', 'geliştir']):
                    recommendation = line.strip()
                    if recommendation and len(recommendation) > 10:  # Kısa önerileri filtrele
                        combined.append(recommendation)

        return list(set(combined))  # Tekrarları kaldır

    async def extract_data(self):
        """PDF'den veri çıkarma işlemi"""
        try:
            with self._pdf_handler() as pdf:
                student_info = {}
                subject_data = {}

                # Öğrenci bilgilerini çıkar
                first_page = pdf.pages[0]
                text = first_page.extract_text()

                # Mevcut koda bu print'leri ekleyin
                print("\n=== PDF'DEN ÇIKARILAN VERİLER ===")
                print("\n1. Öğrenci Bilgileri:")

                for line in text.split('\n'):
                    if "Öğrenci Adı:" in line:
                        student_info['name'] = line.split('Öğrenci Adı:')[1].strip()
                        print(f"İsim: {student_info['name']}")  # Bu print'i ekleyin
                    elif "ÖZEL" in line:
                        student_info['school'] = line.strip()
                        print(f"Okul: {student_info['school']}")  # Bu print'i ekleyin
                    elif "Rapor Tarihi:" in line:
                        student_info['report_date'] = line.split('Rapor Tarihi:')[1].strip()

                # Ders verilerini çıkarmadan önce bu print'i ekleyin
                print("\n2. Bulunan Tablolar:")

                # Ders verilerini çıkar
                for page in pdf.pages:
                    tables = page.extract_tables()
                    print("\n=== PDF'den Çıkarılan Tablolar ===")
                    for table in tables:
                        print("\nTablo İçeriği:")
                        for row in table:
                            print(row)  # Tablonun her satırını görelim

                        # Eğer tablo boş değilse ve en az bir satır içeriyorsa işleme devam et
                        if table and any(row for row in table if any(cell for cell in row if cell)):
                            subject = self._identify_subject(table)
                            if subject:
                                df = self._process_table(table)
                                print(f"\n{subject} Dersi DataFrame:")
                                print(df)
                                subject_data[subject] = df
                    # for table in tables:
                    #     if not table or len(table) < 2:
                    #         continue
                    #
                    #     subject = self._identify_subject(table)
                    #     if subject:
                    #         df = self._process_table(table)
                    #         if not df.empty:
                    #             subject_data[subject] = df

                self.logger.info("Veri çıkarma işlemi başarılı")
                return student_info, subject_data

        except Exception as e:
            self.logger.error(f"Veri çıkarma hatası: {str(e)}")
            raise

    def _identify_subject(self, table):
        """Tablonun hangi derse ait olduğunu belirle"""
        subject_mapping = {
            "TÜRKÇE": "Türkçe",
            "MATEMATİK": "Matematik",
            "FEN BİLİMLERİ": "Fen Bilimleri",
            "İNKILAP TARİHİ": "İnkılap Tarihi",
            "DİN KÜLTÜRÜ": "Din Kültürü",
            "YABANCI DİL": "İngilizce"
        }

        for row in table[:3]:  # İlk birkaç satıra bak
            for cell in row:
                if cell:
                    # Önce İngilizce isimleri Türkçeye çevir
                    for tr, tr_formatted in subject_mapping.items():
                        if tr in str(cell):
                            return tr_formatted
        return None

    def _process_table(self, table):
        """Tabloyu DataFrame'e dönüştür"""
        try:
            headers = table[0]
            df = pd.DataFrame(table[1:], columns=headers)

            # Sütun isimlerini temizle
            df.columns = df.columns.str.strip()

            # Sayısal değerleri dönüştür
            numeric_columns = ['Başarı Oranı %', 'Soru Sayısı']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

            return df

        except Exception as e:
            self.logger.error(f"Tablo işleme hatası: {str(e)}")
            return pd.DataFrame()

    def cleanup(self):
        """Geçici dosyaların temizlenmesi"""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                self.logger.error(f"Geçici dosya silinirken hata: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is not None:
            self.logger.error(f"Hata oluştu: {str(exc_val)}")
            return False
        return True

# async def main():
#     # Komut satırı argümanlarını yapılandır
#     parser = argparse.ArgumentParser(description='Akademik Performans Analiz Sistemi')
#     parser.add_argument('--pdf', type=str, required=True, help='PDF dosyasının yolu')
#     parser.add_argument('--output', type=str, help='Çıktı raporu için klasör yolu')
#     parser.add_argument('--debug', action='store_true', help='Debug modu aktif')
#     args = parser.parse_args()
#
#     # Loglama seviyesini ayarla
#     log_level = logging.DEBUG if args.debug else logging.INFO
#     logging.basicConfig(level=log_level)
#     logger = logging.getLogger(__name__)
#
#     # Çıktı klasörünü hazırla
#     output_dir = Path(args.output) if args.output else Path.cwd() / "reports"
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     try:
#         # ImprovedAcademicAnalyzer'ı context manager ile kullan
#         async with ImprovedAcademicAnalyzer(args.pdf) as analyzer:
#             logger.info("Analiz başlatılıyor...")
#
#             # Veri çıkarma ve validasyon
#             logger.info("PDF'den veri çıkarılıyor...")
#             student_info, subject_data = await analyzer.extract_data()
#
#             # Veri analizi
#             logger.info("Performans analizi yapılıyor...")
#             analysis_results = await analyzer.analyze_performance()
#
#             # Görselleştirme
#             logger.info("Görsel analizler hazırlanıyor...")
#             visualizations = await analyzer.create_visualizations(subject_data)
#
#             # AI tartışması
#             logger.info("AI tartışması başlatılıyor...")
#             discussion_results = await analyzer.conduct_ai_discussion(analysis_results)
#
#             # Rapor oluşturma
#             logger.info("Final rapor hazırlanıyor...")
#             report_path = await analyzer.generate_report(
#                 student_info=student_info,
#                 analysis_results=analysis_results,
#                 visualizations=visualizations,
#                 discussion_results=discussion_results,
#                 output_dir=output_dir
#             )
#
#             logger.info(f"Analiz tamamlandı. Rapor: {report_path}")
#             return report_path
#
#     except FileNotFoundError as e:
#         logger.error(f"PDF dosyası bulunamadı: {e}")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"Beklenmeyen hata: {e}")
#         sys.exit(1)

async def main():
    # PDF dosya yolu
    pdf_path = "./data/KAZANIM_BASARI_DURUMU_12130949.pdf"
    output_dir = Path.cwd() / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ImprovedAcademicAnalyzer'ı context manager ile kullan
        async with ImprovedAcademicAnalyzer(pdf_path) as analyzer:
            print("Analiz başlatılıyor...")

            # Veri çıkarma ve validasyon
            print("PDF'den veri çıkarılıyor...")
            analyzer.student_info, analyzer.subject_data = await analyzer.extract_data()

            # Veri analizi
            print("Performans analizi yapılıyor...")
            analysis_results = await analyzer.analyze_performance()

            # Görselleştirme
            print("Görsel analizler hazırlanıyor...")
            visualizations = await analyzer.create_visualizations(analyzer.subject_data)

            # AI tartışması
            print("AI tartışması başlatılıyor...")
            discussion_results = await analyzer.conduct_ai_discussion(analysis_results)

            # Rapor oluşturma
            print("Final rapor hazırlanıyor...")
            report_path = await analyzer.generate_report(
                student_info=analyzer.student_info,
                analysis_results=analysis_results,
                visualizations=visualizations,
                discussion_results=discussion_results,
                output_dir=output_dir
            )

            print(f"Analiz tamamlandı. Rapor: {report_path}")
            return report_path

    except Exception as e:
        print(f"Hata oluştu: {e}")
        raise


if __name__ == "__main__":
    # Windows'ta asyncio event loop politikasını ayarla
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Ana fonksiyonu çalıştır
    asyncio.run(main())





# def main():
#     # PDF dosya yolu
#     pdf_path = "./data/KAZANIM_BASARI_DURUMU_12130949.pdf"
#
#     try:
#         # Analiz platformunu başlat
#         analyzer = AcademicAnalysisPlatform(pdf_path)
#
#         # Analizi çalıştır
#         report_file = analyzer.run_analysis()
#
#         print("\n=== Analiz Tamamlandı ===")
#         print(f"Rapor dosyası: {report_file}")
#
#     except Exception as e:
#         print(f"\nProgram hatası: {str(e)}")
#         raise


# if __name__ == "__main__":
#     main()
