uv run python scripts/generate.py raw ./raw --split train --version v1        
uv run python scripts/generate.py raw ./raw --split val --version v1        
uv run python scripts/generate.py raw ./raw --split test --version v1        

uv run python scripts/generate.py raw ./raw --split train --version v2        
uv run python scripts/generate.py raw ./raw --split val --version v2        
uv run python scripts/generate.py raw ./raw --split test --version v2        

uv run python scripts/generate.py process ./raw --ignore-warnings --split train --version v1
uv run python scripts/generate.py process ./raw --ignore-warnings --split val --version v1
uv run python scripts/generate.py process ./raw --ignore-warnings --split test --version v1

uv run python scripts/generate.py process ./raw --ignore-warnings --split train --version v2
uv run python scripts/generate.py process ./raw --ignore-warnings --split val --version v2
uv run python scripts/generate.py process ./raw --ignore-warnings --split test --version v2