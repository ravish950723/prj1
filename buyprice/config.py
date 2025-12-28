import os

# === IB / caching ===
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "103"))  # change via env if needed
ALPHA_VANTAGE_API_KEY = "HEFB32P"
# === Labels / thresholds ===
STRONG_BUY_LABEL = {
    "forward_window": 30,
    "min_max_gain_pct": 15.0,
    "max_drawdown_pct": 10.0,
    "min_adx": 20.0,
    "min_volume_weight": 1.0,
}


UPWARD_SIGNAL_WEIGHTS = {
    "smc": 0.05,
    "mean_reversion": 0.03,
    "bullish_engulfing": 0.03,
    "hammer": 0.03,
    "trend_strength": 0.01,

    # thresholds
    "buy_threshold": 0.08,
    "min_signals": 2,
}

# === Macro symbols used for feature engineering ===
# NOTE: Adjust tickers to your actual IBKR symbols if needed.
MACRO_SYMBOLS = {
    "VIX": "VIX",          # Volatility Index (or your VIX ticker)
    "RATE_10Y": "TNX",     # 10Y yield proxy (or ZN/ZN futures equivalent)
    "QQQ": "QQQ",          # Nasdaq-100 ETF
}


symbols = [
    "AAL","AAPL","ABT","ACAD","ACEL","ACIW","ACN","ACHR","ADBE","ADMA","ADP",
    "ADSK","ADT","AFRM","AGX","AI","AIZ","ALAB","ALGN","ALGT","ALK","ALKS",
    "ALNT","ALRM","AME","AMAT","AMBA","AMGN","AMPL","AMTM","AMZN","ANIK",
    "ANIP","AORT","APP","APPN","APLD","APTV","ARGT","ARKK","ARKQ","ARKX",
    "ARM","ARQQ","ASA","ASGN","ASHR","ASTE","ASTS","ATAT","ATEN","ATLC",
    "AUPH","AVAH","AVAV","AVGO","AVNS","AXP", "ASML", "VRT", "BWXT", "CEG",
    "NOK", "MFG", "HYMLF" ,"BA","BAH","BABA","BB","BBAI",
    "BBW","BBY","BCO","BDC","BIDU","BILI","BITB","BITO","BITQ","BITW","BK",
    "BKNG","BKSY","BL","BLX","BMBL","BMY","BOTZ","BPOP","BR","BSWGF","BTBT",
    "BTDR","BTMD","BUG","BWXT","BY","BYD","BZ","BZH","C","CACI","CASH","CASS",
    "CBL","CBOE","CCBG","CCJ","CCK","CCS","CCSI","CDNS","CEG","CELH","CEVA",
    "CF","CFLT","CGNX","CHH","CHKP","CI","CIBR","CIFR","CL","CLSK","CMCSA",
    "CMRE","CNC","CNXN","COIN","COLL","COMP","CONI","COR","CORZ","CPA","CPAY",
    "CPER","CPNG","CQQQ","CRDO","CRC","CRNC","CRPT","CRM","CRS","CRSR","CRUS",
    "CRVL","CRWD","CSCO","CSGS","CSV","CTEV","CTLP","CTVA","CVCO","CVLT",
    "CVS","CVV","CW","CXM","DAL","DAPP","DASH","DCBO","DCI","DDD","DD","DELL",
    "DES","DFH","DGII","DHI","DIS","DJT","DLR","DLX","DMAT","DOMO","DRI","DRS",
    "DUK","DVAX","DXCM","DXYZ","EA","EBAY","EBF","ECL","EDU","EEM","EH","EHAB",
    "ELMD","EMR","ENS","ENTA","EQIX","ESCA","ESLT","ESPO","ESTC","ETD",
    "ETSY","EVGO","EVTC","EXLS","EXOD","EXPE","EXPI","F","FBP","FDIG","FET",
    "FFIV","FG","FIS","FLXS","FLYW","FMC","FOUR","FRSH","FSLR",
    "FTDR","FTNT","FTXL","FUNC","FUBO","GAMR","GBTC","GBTG","GCBC","GD","GDDY",
    "GDX","GDXJ","GE","GEN","GENC","GFF","GFI","GHC","GILD","GLBE","GLD","GM",
    "GOOG","GOOGL","GOTU","GPN","GRAB","GRBK","GRC","GRMN","GRND","GRPN","GRRR",
    "GS","GSAT","GSK","GTX","GWRE","GXO","H","HACK","HAL","HAS","HCAT",
    "HD","HEI","HERO","HGV","HIG","HII","HIMX","HIVE","HLIT","HLT","HON",
    "HOOD","HOUS","HOV","HPE","HPQ","HRB","HRMY","HRTG","HSTM","HUBS",
    "HUYA","HUT","HWBK","IAS","IBM","IBEX","IBIT","ICE","IDT","IJT","III",
    "IMMR","INDA","INDS","INFY","INNV","INOD","INSE","INTC","INTU",
    "IONQ","IPO","IPGP","IQ","IREN","IRDM","IRMD","IRTC","ISRG","IT","IYR",
    "JAKK","JAMF","JBI","JBL","JBLU","JD","JMIA","JNJ","JOBY","JOYY","KAR",
    "KBH","KBR","KC","KD","KELYA","KGC","KMT","KOPN","KR","KRNT","KTOS",
    "KWEB","LAUR","LDOS","LECO","LEN","LEU","LEVI","LGIH","LGND","LHX","LI",
    "LIND","LITE","LIVN","LKQ","LMAT","LOGI","LOPE","LRN","LMT","LSPD","LTBR",
    "LUNR","LUV","LVS","M","MA","MAMA","MANH","MARA","MASI","MAT","MAX","MDB",
    "MD","MDT","MELI","META","MFIN","MGEE","MISL","MLI","MNDY","MO",
    "MOMO","MOV","MP","MPAA","MRCY","MRVL","MS","MSFT","MSI","MSTR","MU",
    "MYE","NAIL","NATH","NATR","NBIS","NCNO","NDAQ","NDSN","NEE","NEM","NET",
    "NFLX","NFG","NGVT","NHC","NICE","NIO","NKE","NLR","NNI","NNOX","NOC",
    "NOVT","NSANY","NTAP","NTCT","NTES","NTNX","NU","NUKZ","NVDA","NVGS",
    "NXST","OBDC","OC","OFG","OIS","OKLO","OKTA","OMCL","OPEN","ORCL",
    "ORGO","OSBC","OSPN","OTEX","OUST","OVLY","OVV","PAG","PANW","PAR","PATH",
    "PAY","PAYO","PBYI","PCG","PDD","PEB","PEBK","PEGA","PFE","PHM","PICK",
    "PINS","PL","PLTK","PLTR","PLUS","PLUG","PLXS","PRDO","PRGS","PRLB",
    "PRVA","PSN","PSTG","PTCT","PXH","PYPL","QBTS","QCOM","QNCCF","QNTM",
    "QS","QTUM","QTWO","QUBT","QLYS","RAMP","RBLX","RBRK","RBCAA","RBKB",
    "RCL","RDNT","RDVT","RDW","REMX","REPX","RGTI","RIGL","RIOT","RKLB",
    "ROBO","ROKU","RPAY","RPD","RR","RRBI","RUM","RVMD","RXT","RZLV","RZV",
    "S","SABR","SAIC","SAMG","SATL","SATS","SCHD","SCHH","SCHA","SCSC",
    "SDHC","SE","SERV","SETM","SFST","SGC","SH","SHLD","SHOP","SIRI","SIGA",
    "SIGI","SLV","SLYG","SMCI","SMLR","SMR","SNAP","SNOW","SNA","SNCY",
    "SOFI","SOHU","SONY","SOUN","SPFI","SPIR","SPOT","SPT","SSNC",
    "SSO","STCE","STNE","STRT","STT","SUPN","SWX","SXC","SYF","SYK","SYM",
    "SYNA","T","TAL","TALO","TBCH","TBPH","TCMD","TDC","TDOC","TDY","TECL",
    "TEM","TENB","TER","THFF","THNQ","TIGO","TK","TMHC","TME","TMUS","TNC",
    "TNDM","TNK","TOL","TPH","TRI","TRIP","TRU","TSAT","TSBK","TSLA","TSM",
    "TTMI","TTWO","UAL","U","ULCC","UNH","UNTY","URA","URNM","USAR","USCB",
    "USD","UTHR","UTMD","UVE","V","VAC","VALU","VB","VBK","VEA","VEEV",
    "VICI","VIPS","VKTX","VLRS","VNDA","VNT","VRSK","VRNS","VRT","VST",
    "VTEX","VTWO","VZ","WAB","WAY","WB","WCC","WDAY","WEYS","WGMI","WIT",
    "WIX","WK","WRLD","WSBF","WTS","WULF","XME","XMTR","XNET","XOVR",
    "XPER","XPRO","XRAY","XYF","YALA","Z","ZBRA","ZEPP","ZH","ZM","ZS",
    "ZTEK"
]


symbol_to_sector = {'AMBA': 'Semiconductor', 'ARQQ': 'Tech', 'ASTS': 'Satellite', 'BBAI': 'AI', 'BTBT': 'CryptoMining',
                    'CIFR': 'CryptoMining', 'EXPE': 'TravelTech',
                    'INTC': 'Semiconductor', 'IONQ': 'Quantum', 'MP': 'RareEarth', 'NDAQ': 'Exchange',
                    'NNE': 'NuclearETF', 'OKLO': 'Nuclear', 'QBTS': 'Quantum',
                    'QNTM': 'Quantum', 'QS': 'Battery', 'QUBT': 'Quantum', 'RBRK': 'Cybersecurity', 'REPX': 'OilGas',
                    'RGTI': 'Quantum', 'RKLB': 'Aerospace',
                    'RNGR': 'OilServices', 'SMR': 'Nuclear', 'SOUN': 'AI', 'TNL': 'TravelLeisure', 'USAR': 'Other',
                    'FTNT': 'Cybersecurity', 'ARGT': 'ArgentinaETF',
                    'ARKQ': 'InnovationETF', 'ARKK': 'InnovationETF', 'ARKX': 'SpaceETF', 'BITQ': 'CryptoETF',
                    'BUG': 'CybersecurityETF', 'CPER': 'CopperETF',
                    'CRPT': 'CryptoETF', 'DRAG': 'DefenseETF', 'DXYZ': 'AIETF', 'ESPO': 'GamingETF', 'GLD': 'GoldETF',
                    'HACK': 'CybersecurityETF', 'HERO': 'GamingETF',
                    'IBIT': 'BitcoinETF', 'INDS': 'REITETF', 'MISL': 'DefenseETF', 'NUKZ': 'NuclearETF',
                    'PPA': 'DefenseETF', 'PXH': 'EmergingMarketsETF',
                    'QTUM': 'AIETF', 'SHLD': 'DefenseETF', 'SLV': 'SilverETF', 'WGMI': 'CryptoMiningETF',
                    'XOVR': 'DisruptiveETF', 'COIN': 'Bitcoin', 'MSTR': 'Bitcoin',
                    'RIOT': 'Bitcoin', 'MARA': 'Bitcoin', 'HUT': 'Bitcoin', 'CLSK': 'Bitcoin', 'CORZ': 'Bitcoin',
                    'STCE': 'BitcoinETF', 'GBTC': 'BitcoinETF',
                    'BITO': 'BitcoinETF', 'BITB': 'BitcoinETF', 'BITW': 'BitcoinETF', 'BABA': 'ChinaStock',
                    'JD': 'ChinaStock', 'PDD': 'ChinaStock', 'NIO': 'ChinaStock',
                    'XPEV': 'ChinaStock', 'LI': 'ChinaStock', 'TCEHY': 'ChinaStock', 'BIDU': 'ChinaStock',
                    'KWEB': 'ChinaETF', 'CQQQ': 'ChinaETF', 'FXI': 'ChinaETF',
                    'MCHI': 'ChinaETF', 'ASHR': 'ChinaETF', 'NAIL': 'HomebuildersBull', 'SOXL': 'SemiconductorsBull',
                    'USD': 'SemiconductorsBull', 'TECL': 'TechnologyBull',
                    'TQQQ': 'TechnologyBull', 'SSO': 'SP500Bull', 'SH': 'SP500Bear', 'QID': 'NasdaqBear',
                    'NVDL': 'SingleStock_NVDA_Bull', 'NVDU': 'SingleStock_NVDA_Bull',
                    'CONL': 'SingleStock_COIN_Bull', 'CONI': 'SingleStock_COIN_Bear', 'MSTU': 'SingleStock_MSFT_Bull',
                    'FIAT': 'MacroCurrency', 'GOOG': 'BigTech', 'MSFT': 'BigTech',
                    'IBM': 'BigTech', 'NVDA': 'BigTech', 'SPRX': 'InnovationETF', 'NOC': 'AeroDefense',
                    'BA': 'AeroDefense', 'LMT': 'AeroDefense', 'RTX': 'AeroDefense',
                    'LHX': 'AeroDefense', 'HEI': 'AeroDefense', 'HXL': 'AeroDefense', 'ESLT': 'AeroDefense',
                    'AVAV': 'AeroDefense', 'KTOS': 'AeroDefense', 'PSN': 'AeroDefense',
                    'ERJ': 'AeroDefense', 'BKSY': 'SpaceTech', 'PL': 'SpaceTech', 'SATL': 'SpaceTech',
                    'SPIR': 'SpaceTech', 'RDW': 'SpaceTech', 'SATS': 'SatCom', 'IRDM': 'SatCom',
                    'SIRI': 'SatCom', 'GSAT': 'SatCom', 'VSAT': 'SatCom', 'ADBE': 'Software', 'ADP': 'ITServices',
                    'ADT': 'Industrials', 'ALEX': 'REITETF', 'ALKS': 'Biotech',
                    'APTV': 'Industrials', 'BK': 'Banks', 'BKNG': 'ConsumerDiscretionary', 'BLX': 'Banks',
                    'BWA': 'Industrials', 'BYD': 'ConsumerDiscretionary', 'CBL': 'REITETF',
                    'COR': 'HealthcareDistributors', 'CSGS': 'Tech', 'CTVA': 'Materials', 'CVS': 'HealthcareProviders',
                    'DD': 'Materials', 'DDS': 'ConsumerDiscretionary',
                    'DUK': 'Utilities', 'EPR': 'REITETF', 'FET': 'Energy', 'GFF': 'Industrials', 'HCKT': 'ITServices',
                    'HNI': 'Industrials', 'KELYA': 'Industrials', 'KR':
                        'ConsumerStaples', 'MANH': 'Software', 'MAX': 'Tech', 'MCK': 'HealthcareDistributors',
                    'MD': 'HealthcareProviders', 'MGEE': 'Utilities', 'MO': 'ConsumerStaples',
                    'MTCH': 'Communications', 'NAGE': 'Tech', 'NATR': 'ConsumerStaples', 'NEU': 'Materials',
                    'NFG': 'Utilities', 'NFLX': 'Communications', 'NGVT': 'Materials', 'NLOP':
                        'REITETF', 'NRC': 'HealthcareProviders', 'OVV': 'Energy', 'PBYI': 'Biotech', 'PCG': 'Utilities',
                    'PTCT': 'Biotech', 'QLYS': 'CybersecurityETF', 'RRBI': 'Banks',
                    'SFD': 'ConsumerStaples', 'SPOK': 'Communications', 'SWX': 'Utilities', 'SXC': 'Materials',
                    'SYY': 'ConsumerStaples', 'TALO': 'Energy', 'TEL': 'Industrials',
                    'THFF': 'Banks', 'TIGO': 'Communications', 'TMUS': 'Communications', 'TSBK': 'Banks',
                    'VICI': 'REITETF', 'WAY': 'HealthcareProviders', 'WEYS': 'ConsumerDiscretionary',
                    'AMD': 'AIStock', 'GOOGL': 'AIStock', 'AAPL': 'AIStock', 'META': 'AIStock', 'AMZN': 'AIStock',
                    'TSLA': 'AIStock', 'AI': 'AIStock', 'PLTR': 'AIStock',
                    'SNOW': 'AIStock', 'PATH': 'AIStock', 'CRWV': 'AIStock', 'ROBO': 'AIETF', 'BOTZ': 'AIETF',
                    'THNQ': 'AIETF', 'CRWD': 'CyberSecurityStock', 'ZS': 'CyberSecurityStock',
                    'S': 'CyberSecurityStock', 'PANW': 'CyberSecurityStock', 'CHKP': 'CyberSecurityStock',
                    'CYBR': 'CyberSecurityStock', 'CIBR': 'CyberSecurityETF', 'EA': 'GamingStock',
                    'TTWO': 'GamingStock', 'RBLX': 'GamingStock', 'SONY': 'GamingStock', 'NERD': 'GamingETF',
                    'GAMR': 'GamingETF', 'NNXPF': 'Graphene', 'GMGMF': 'Graphene',
                    'BSWGF': 'Graphene', 'ZTEK': 'Graphene', 'FGPHF': 'Graphene', 'HDGHF': 'Graphene',
                    'VRSRF': 'Graphene', 'GTI': 'Graphene', 'CVV': 'GrapheneEquipment',
                    'DMAT': 'GrapheneETF', 'SETM': 'GrapheneETF', 'GDX': 'MetalETF', 'GDXJ': 'MetalETF',
                    'PICK': 'MetalETF', 'XME': 'MetalETF', 'URA': 'MetalETF', 'DBB': 'MetalETF',
                    'REMX': 'MetalETF', 'LTBR': 'Nuclear', 'BWXT': 'Nuclear', 'LEU': 'Nuclear', 'CCJ': 'Nuclear',
                    'CEG': 'Nuclear', 'VST': 'Nuclear', 'NLR': 'NuclearETF'
                    }

sector_etfs = {'Tech': 'XLK', 'AI': 'DXYZ', 'Quantum': 'QTUM', 'CryptoMining': 'WGMI', 'CryptoETF': 'BITQ',
               'DefenseETF': 'PPA', 'GamingETF': 'ESPO', 'GoldETF': 'GLD', 'SilverETF': 'SLV', 'SpaceETF': 'ARKX',
               'REITETF': 'INDS', 'EmergingMarketsETF': 'PXH', 'InnovationETF': 'ARKQ', 'ArgentinaETF': 'ARGT',
               'DisruptiveETF': 'XOVR', 'AIETF': 'QTUM', 'Semiconductor': 'SOXX', 'TravelTech': 'AWAY',
               'TravelLeisure': 'PEJ', 'Cybersecurity': 'HACK', 'OilGas': 'XOP', 'OilServices': 'OIH',
               'Nuclear': 'NUKZ', 'Insurtech': 'KIE', 'RareEarth': 'REMX', 'Battery': 'LIT', 'Aerospace': 'ITA',
               'Exchange': 'IAI', 'NuclearETF': 'NUKZ', 'Other': 'SPY', 'Satellite': 'ARKX', 'CybersecurityETF': 'BUG',
               'CopperETF': 'CPER', 'BitcoinETF': 'IBIT', 'Bitcoin': 'STCE', 'ChinaETF': 'KWEB',
               'HomebuildersBull': 'XHB', 'SemiconductorsBull': 'SOXX', 'TechnologyBull': 'QQQ', 'SP500Bull': 'SPY',
               'SP500Bear': 'SPY', 'NasdaqBear': 'QQQ', 'SingleStock_NVDA_Bull': 'NVDA',
               'SingleStock_COIN_Bull': 'COIN', 'SingleStock_COIN_Bear': 'COIN', 'SingleStock_MSFT_Bull': 'MSFT',
               'MacroCurrency': 'UUP', 'BigTech': 'XLK', 'QuantumETF': 'QTUM', 'AeroDefense': 'ITA', 'SpaceTech': 'UFO',
               'SatCom': 'UFO', 'ThematicETF': 'DXYZ', 'ARated': None, 'CyberSecurityETF': 'CIBR', 'Graphene': 'DMAT',
               'GrapheneEquipment': 'SOXX', 'GrapheneETF': 'DMAT', 'Graphite': 'SETM', 'MetalETF': 'GDX'}

sector_map = {'Bitcoin': 'finance', 'BitcoinETF': 'etf', 'CryptoMining': 'finance', 'CryptoETF': 'etf',
              'Quantum': 'tech', 'CopperETF': 'commodity', 'TravelTech': 'consumer', 'Semiconductor': 'tech',
              'Nuclear': 'energy', 'AI': 'tech', 'TravelLeisure': 'consumer', 'Other': 'broad', 'Cybersecurity': 'tech'}

symbol_to_industry = {}
# --- VIX-adaptive signal tuning ---
# These multipliers let you automatically get more conservative in high-volatility regimes
# without retraining the ML model. Keys are matched by substring search on VIX_regime, e.g. "HIGH".
VIX_ADAPTIVE_WEIGHTS = {
    "LOW":    {"weight_mult": 1.00, "buy_threshold_mult": 1.00, "min_signals_add": 0},
    "NORMAL": {"weight_mult": 1.00, "buy_threshold_mult": 1.00, "min_signals_add": 0},
    "MEDIUM": {"weight_mult": 0.90, "buy_threshold_mult": 1.10, "min_signals_add": 0},
    "HIGH":   {"weight_mult": 0.75, "buy_threshold_mult": 1.20, "min_signals_add": 1},
}

# Optional: weights used by symbol_analysis._calc_signal_score (kept separate from UPWARD_SIGNAL_WEIGHTS)
RULE_SIGNAL_WEIGHTS = {
    "vwap_support": 0.15,
    "ema_uptrend": 0.15,
    "macd_cross": 0.10,
    "rsi_state": 0.10,
    "near_support": 0.10,
    "volume_surge": 0.10,
    "darvas_signal": 0.20,
    "smc_breakout": 0.10,
}
