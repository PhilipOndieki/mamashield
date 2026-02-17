# 🛡️ MamaShield: The Digital Shield

**GEGIS Hackathon 2026 - Track 5 Submission**

An interactive dashboard analyzing whether maternal education and information access can protect mothers and children from health risks in geographically disadvantaged areas of Kenya.

---

## 🎯 Research Question

**"Does maternal education and information access reduce health risks for mothers living in remote or high-malaria zones?"**

This dashboard provides spatial evidence to answer the "Why" question: *Why do some mothers in risky zones have better health outcomes than others in safer zones?*

---

## 📊 Key Features

### Interactive Dashboard Components

1. **Overview Tab** - Key metrics and data distribution
2. **County Map** - Geographic visualization of shield effect by county
3. **Shield Effect Analysis** - Core comparison of educated vs uneducated mothers in high-risk zones
4. **Deep Dive** - Scatter plots and detailed group statistics
5. **Key Insights** - Auto-generated findings and actionable recommendations

### Interactive Filters

- **Education Threshold Slider** - Define what constitutes "educated"
- **Risk Type Selector** - Analyze combined risk, malaria only, or remoteness only
- **County Filter** - Focus analysis on specific counties

---

## 🔬 Methodology

### Composite Indices

**Education Score (0-10):**
- Education level (0-5 points)
- Mobile phone ownership (+2 points)
- TV ownership (+2 points)
- Radio ownership (+1 point)

**Risk Score (0-10):**
- Travel time to nearest city (normalized)
- Malaria prevalence (normalized)

**Categories:**
- Low: 0-3.33
- Medium: 3.34-6.66
- High: 6.67-10

### Shield Effect Calculation

**Shield Effect** = (Survival rate: Educated + High-Risk) - (Survival rate: Uneducated + High-Risk)

- Positive value = Education provides protection
- Higher values = Stronger protective effect

---

## 💡 Key Findings

### 1. The Shield Exists
Educated mothers in high-risk zones show **significantly higher** child survival rates compared to uneducated mothers in the same zones.

### 2. The Shield Varies
Shield effectiveness differs by county - some regions show education making a 20%+ difference, while others show minimal impact.

### 3. Education Can Overcome Geography
In many cases, educated mothers in dangerous zones achieve **equal or better** outcomes than uneducated mothers in safe zones.

### 4. Priority Intervention Areas Identified
Counties with high geographic risk but low education levels represent the greatest opportunity for impact.

---

## 📁 Project Structure

```
mamashield/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Data loading and merging with caching
├── data_processor.py      # Index calculation and aggregations
├── visualizations.py      # Plotly chart generation
├── utils.py              # Helper functions and insights
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- Ubuntu 20.04 or later
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download Project

```bash
# Navigate to your working directory
cd ~

# If you have the project files, ensure they're in a folder called 'mamashield'
# Your directory structure should be:
# ~/mamashield/          (application files)
# ~/hackathon_data/      (dataset files)
```

### Step 2: Install System Dependencies

```bash
# Update package list
sudo apt update

# Install Python and pip if not already installed
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies for GeoPandas
sudo apt install gdal-bin libgdal-dev libspatialindex-dev -y
```

### Step 3: Create Virtual Environment

```bash
# Navigate to project directory
cd ~/mamashield

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv)
```

### Step 4: Install Python Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# This will install:
# - streamlit==1.31.0
# - pandas==2.1.4
# - geopandas==0.14.2
# - plotly==5.18.0
# - numpy==1.26.3
# - scipy==1.11.4
```

### Step 5: Verify Data Files

Ensure your data files are in the correct location:

```bash
# Check that data directory exists
ls ~/hackathon_data/

# You should see:
# - kenya_dhs_dataset_complete.csv
# - kenya_dhs_covariates.csv
# - kenya_dhs_dataset_gps.geojson
# - DHS recode manual.pdf
```

### Step 6: Run the Application

```bash
# Make sure you're in the mamashield directory
cd ~/mamashield

# Make sure virtual environment is activated
source venv/bin/activate

# Run Streamlit
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

---

## 🖥️ Usage

### Basic Navigation

1. **Use the sidebar** to adjust filters:
   - Adjust education threshold to see how definition changes results
   - Switch between risk types to analyze different threats
   - Filter to specific counties for detailed analysis

2. **Explore the tabs** from left to right:
   - Start with Overview to understand the data
   - Check County Map to see geographic patterns
   - Analyze Shield Effect for core findings
   - Deep Dive for detailed statistics
   - Review Key Insights for actionable recommendations

### Tips for Best Results

- Start with default settings to see overall patterns
- Use education threshold slider to find optimal intervention point
- Compare "Malaria Risk Only" vs "Remoteness Only" to see which matters more
- Look for counties in the "success stories" section to identify best practices

---

## 📈 Data Sources

**Primary Data:**
- Kenya DHS 2022 Births Recode (77,381 births)
- GPS Covariates (environmental data for 1,691 clusters)
- Geographic coordinates and county boundaries

**Variables Used:**
- **Education:** V106 (education level), V107 (years of education)
- **Information Access:** V119 (mobile), V120 (TV), V121 (radio)
- **Health Outcomes:** B5 (child alive), B8 (age at death)
- **Environmental Risks:** Travel times, malaria prevalence, rainfall, wet days
- **Geographic:** County names, coordinates, urban/rural classification

---

## 🛠️ Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'streamlit'"**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

**2. "Error loading data: FileNotFoundError"**
```bash
# Check data path in app.py (default is ../hackathon_data)
# Or move data files to correct location:
mkdir -p ~/hackathon_data
# Copy your data files there
```

**3. "GDAL/GeoPandas installation error"**
```bash
# Install system dependencies first
sudo apt install gdal-bin libgdal-dev libspatialindex-dev -y

# Then install geopandas
pip install geopandas==0.14.2
```

**4. Application is slow**
```bash
# This is normal on first load as it caches data
# Subsequent interactions will be much faster
# Wait 30-60 seconds for initial data loading
```

---

## 📊 Performance Notes

- **Initial Load:** 30-60 seconds (one-time data loading and caching)
- **Filter Changes:** <2 seconds (cached data)
- **Memory Usage:** ~2-3 GB RAM
- **Recommended:** 4+ GB RAM, modern processor

---

## 🎨 Competition Category

**Interactive Dashboard** - Built for decision-makers to slice data by county, education level, and risk factors to identify intervention priorities.

---

## 👥 Credits

**Developed by:** Philolo  
**Competition:** GEGIS Hackathon 2026  
**Track:** Track 5 - The Digital Shield  
**Date:** February 2026

---

## 📝 License

This project was developed for the GEGIS Hackathon 2026. Dataset provided by hackathon organizers.

---

## 🙏 Acknowledgments

- GEGIS Hackathon organizers for providing the dataset
- DHS Program for the Kenya 2022 survey data
- Anthropic's Claude for development assistance

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all data files are in the correct location
3. Ensure virtual environment is activated
4. Check that all dependencies installed successfully

---

**Built with:** Python, Streamlit, Pandas, GeoPandas, Plotly  
**Development Time:** 12 hours  
**Status:** Production-ready MVP
