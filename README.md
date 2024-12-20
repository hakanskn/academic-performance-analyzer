# Academic Performance Analyzer

**Academic Performance Analyzer**, a tool designed to analyze academic performance, generate insightful visualizations, and create comprehensive reports with AI-powered recommendations.

---

**Academic Performance Analyzer**, akademik performansı analiz etmek, görselleştirmeler oluşturmak ve yapay zeka destekli önerilerle kapsam lı raporlar hazırlamak için geliştirilmiş bir araçtır.

---

## Features / Özellikler

- **PDF Parsing**: Extracts student and subject data from PDF files.
- **Performance Analysis**: Provides a detailed analysis of success rates, strong topics, and areas for improvement.
- **Visualizations**: Generates clear and informative charts for data visualization.
- **AI-Powered Insights**: Utilizes OpenAI and Anthropic APIs to simulate discussions and provide expert recommendations.
- **Report Generation**: Creates professional PDF reports for academic performance evaluation.

---

- **PDF İşleme**: Öğrenci ve ders verilerini PDF dosyalarından çıkarır.
- **Performans Analizi**: Başarı oranları, güçlü konular ve geliştirilmesi gereken alanlar için detaylı analiz sağlar.
- **Görselleştirme**: Verileri grafiklerle anlaşılır hale getirir.
- **Yapay Zeka Destekli İçgörüler**: OpenAI ve Anthropic API’leri kullanılarak uzman görüşleri simüle edilir.
- **Rapor Oluşturma**: Akademik performansı detaylı bir şekilde özetleyen PDF raporlar oluşturur.

---

## Requirements / Gereksinimler

- Python 3.8+
- Required libraries (install with `pip`):
  - `openai`, `anthropic`, `pdfplumber`, `pandas`, `matplotlib`, `seaborn`, `fpdf`, `aiohttp`, `tenacity`

---

- Python 3.8+
- Gerekli kütüphaneler (şu komutla yüklenebilir: `pip`):
  - `openai`, `anthropic`, `pdfplumber`, `pandas`, `matplotlib`, `seaborn`, `fpdf`, `aiohttp`, `tenacity`

---

## Installation / Kurulum

1. Clone the repository:

   ```bash
   git clone https://github.com/hakanskn/academic-performance-analyzer.git
   cd academic-performance-analyzer
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. API anahtarlarınızı `config.json` dosyasına ekleyin:
   ```json
   {
       "openai": "your-openai-api-key",
       "anthropic": "your-anthropic-api-key"
   }
   ```

---

1. Depoyu klonlayın:

   ```bash
   git clone https://github.com/hakanskn/academic-performance-analyzer.git
   cd academic-performance-analyzer
   ```

2. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

3. API anahtarlarınızı ortam değişkenleri olarak ayarlayın:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

---

## Usage / Kullanım

1. Prepare a PDF file with academic data.
2. Run the analyzer:
   ```bash
   python academic_insight_collaborative.py --pdf /path/to/pdf --output /path/to/output
   ```
3. The tool will:
   - Extract data from the PDF.
   - Perform analysis and create visualizations.
   - Simulate expert discussions for actionable insights.
   - Generate a comprehensive report in the specified output directory.

---

1. Analiz edilecek PDF dosyasını hazırlayın.
2. Analiz aracını çalıştırın:
   ```bash
   python academic_insight_collaborative.py --pdf /path/to/pdf --output /path/to/output
   ```
3. Araç aşağıdaki işlemleri yapacaktır:
   - PDF’den verileri çıkarır.
   - Analiz yapar ve grafikler oluşturur.
   - Eyleme geçirilebilir öneriler için uzman görüşlerini simüle eder.
   - Belirtilen dizine kapsam lı bir rapor oluşturur.

---

## Contribution / Katkı Sağlama

1. Fork the repository.
2. Make your changes.
3. Submit a pull request.

---

1. Depoyu fork edin.
2. Değişikliklerinizi yapın.
3. Bir pull request gönderin.

---
