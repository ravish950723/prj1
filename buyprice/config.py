import os

# === IB / caching ===
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "103"))  # change via env if needed

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
    "mean_rev": 0.03,
    "bullish_engulfing": 0.03,
    "hammer": 0.03,
    "trend_strength": 0.01,
    "buy_threshold": 0.08,
    "min_signals": 2,
}

symbols = [
    # Individual Stocks
    "AAL", "AAPL", "ACIW", "ACN", "ACHR", "ADBE", "AGX", "AI", "ALNT", "AMD",
    "AMBA", "AMGN", "AMPL", "AMZN", "APLD", "APP", "ARM", "ARKQ", "ASA", "ASGN",
    "ASTS", "ATAT", "AVAV", "AVGO", "BA", "BAH", "BABA", "BBW", "BBAI", "BIDU",
    "BILI", "BITB", "BITO", "BITQ", "BITW", "BK", "BKSY", "BMBL", "BOTZ", "BTBT",
    "BTDR", "BUG", "BZ", "CFLT", "CCJ", "CCS", "CEG", "CELH", "CHH", "CHKP", "CIBR",
    "CIFR", "CLSK",  "CNC", "COIN", "COMP", "CORZ", "CPA", "CPAY", "CPER",
    "CPNG", "CRPT", "CRM", "CRWD", "CVCO", "CVV", "CYBR", "CW", "DASH", "DCBO",
    "DFH", "DGII", "DMAT", "DOMO", "DXYZ", "DXPE", "EA", "EBAY", "ERJ", "ESLT",
    "ESPO", "ESTC", "ETSY", "EXOD", "EXPE", "F", "FDIG", "FICO", "FI", "FIS",
    "FLYW",  "FMC", "FOUR", "FRSH", "FTNT", "GAMR", "GBTC", "GBTG", "GDDY",
    "GFI", "GLBE", "GLD", "GOOG", "GOOGL", "GRND", "GRPN", "GRRR", "GTI", "GWRE",
    "HACK", "HAS", "HCAT", "HEI", "HERO", "HII", "HLIT", "HOOD", "HOV", "HOUS",
    "HRTG", "HIVE", "HUBS", "HUT", "IBIT", "IBM", "INDS", "INFY", "INOD", "INTC",
    "IONQ", "IREN", "IRDM", "III", "JNJ", "JOBY", "JOYY", "JBLU", "KBR", "KGC",
    "KOPN", "KTOS", "LDOS", "LEU", "LEVI", "LGIH", "LHX", "LMT", "LTBR",
    "LSPD", "LUNR", "M", "MAMA", "MARA", "MDB", "META", "MISL", "MP", "MRVL",
    "MRCY", "MSFT", "MU", "NDAQ", "NBIS", "NCNO", "NEM", "NERD", "NET", "NEE",
    "NFLX", "NLR",  "NU", "NUKZ", "OBDC", "OKLO",
    "OPEN", "ORCL", "OTEX", "OUST", "PBYI", "PANW", "PAR", "PATH", "PPA", "PSTG",
    "PLUS", "PL", "PLTK", "PLTR", "PRGS", "PSN", "PXH", "QBTS", "QNTM", "QTUM",
    "QTWO", "QUBT", "QS", "QURE", "RDVT", "RBLX", "RBRK", "REPX", "RIOT", "RKLB",
    "RGTI", "ROBO", "ROKU", "RPAY", "RPD", "RR", "RVMD", "RUM", "RZLV", "S",
    "SABR", "SATL", "SATS", "SE", "SERV", "SETM", "SHLD", "SHOP", "SIRI", "SLV",
    "SMCI", "SMR", "SMLR", "SNCY", "SNAP", "SNOW", "SOFI", "SOUN", "SPNS", "SPOT",
    "SPRX", "SPT", "SPIR", "SPRX", "SPOT", "SPRX", "SPRX", "SPRX", "STCE",
    "STNE", "STRL", "SYM", "TDOC", "TDC", "TDY", "TEM", "TENB", "TER", "THNQ",
    "TMC", "TME", "TNDM", "TNL",  "TRU", "TSAT", "TSLA", "TSM", "TTMI",
    "TTWO", "UAL", "U", "ULCC", "UNH", "URA", "URNM", "USD", "USAR", "VAC",
    "VEEV", "VIPS", "VRNS", "VRSK", "VRT", "VST", "VTEX", "WB", "WGMI", "WBTN",
    "WULF", "XNET", "XOVR", "XPER", "XYF", "ZTEK", "ZS"
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
                    'CEG': 'Nuclear', 'VST': 'Nuclear', 'NLR': 'NuclearETF',
                    'URNM': 'NuclearETF', 'NAIL': 'HomebuildersBull', 'USD': 'SemiconductorsBull',
                    'TECL': 'TechnologyBull', 'TQQQ': 'TechnologyBull', 'SSO': 'SP500Bull',
                    'SOXL': 'SemiconductorsBull', 'NVDL': 'SingleStock_NVDA_Bull', 'NVDU': 'SingleStock_NVDA_Bull',
                    'CONL': 'SingleStock_COIN_Bull', 'SH': 'SP500Bear', 'QID': 'NasdaqBear',
                    'CONI': 'SingleStock_COIN_Bear', 'MSTU': 'SingleStock_MSFT_Bull', 'FIAT': 'MacroCurrency',
                    'IONX': 'SingleStock_IONQ_Bull'
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
