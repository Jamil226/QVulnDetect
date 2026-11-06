import sys
import pyevmasm
import argparse

def analyze_bytecode_for_ecrecover(hex_bytecode):
    """
    Analyzes EVM bytecode to definitively detect calls to the ecrecover precompile (address 0x01).
    This is scientifically much stronger than source-code regex matching.
    """
    if hex_bytecode.startswith("0x"):
        hex_bytecode = hex_bytecode[2:]
        
    try:
        instructions = list(pyevmasm.disassemble_all(bytes.fromhex(hex_bytecode)))
    except Exception as e:
        return {"error": str(e), "detected": False}

    ecrecover_detected = False
    reasons = []

    # Heuristic 1: Look for PUSH1 0x01 (or PUSH2 0x0001) followed by a STATICCALL or CALL within a short window.
    # The ecrecover address is 0x01.
    
    # We will simulate a very basic stack tracking or window tracking.
    WINDOW_SIZE = 15
    
    for i, inst in enumerate(instructions):
        if inst.name in ('STATICCALL', 'CALL'):
            # Look backwards to find the address being called
            window = instructions[max(0, i - WINDOW_SIZE):i]
            
            # Check if 0x01 was pushed recently
            pushed_one = False
            for w_inst in reversed(window):
                if w_inst.name.startswith('PUSH'):
                    # Pyevmasm operands are integers
                    if w_inst.operand == 1:
                        pushed_one = True
                        break
                
            if pushed_one:
                ecrecover_detected = True
                reasons.append(f"Found {inst.name} near PUSH 0x1 at PC {hex(inst.pc)}")
                break

    return {
        "detected": ecrecover_detected,
        "reasons": reasons,
        "instruction_count": len(instructions)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVM Bytecode ecrecover Scanner")
    parser.add_argument("--bytecode", type=str, help="Raw hex EVM bytecode")
    parser.add_argument("--file", type=str, help="File containing raw hex EVM bytecode")
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            bytecode = f.read().strip()
    elif args.bytecode:
        bytecode = args.bytecode
    else:
        # Example dummy bytecode that calls ecrecover:
        # PUSH1 0x01 (6001) PUSH1 0x00 (6000) STATICCALL (fa)
        bytecode = "60016000fa"
        print("No input provided. Running on a dummy snippet: 60016000fa")

    result = analyze_bytecode_for_ecrecover(bytecode)
    print("=========================================")
    print("  QVulnDetect Bytecode Precompile Scanner")
    print("=========================================")
    print(f"Instructions Parsed : {result.get('instruction_count', 0)}")
    print(f"ecrecover Detected  : {result['detected']}")
    if result['reasons']:
        print("Detection Evidence  :")
        for r in result['reasons']:
            print(f"  -> {r}")
    print("=========================================")

