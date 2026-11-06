import os
import argparse
from web3 import Web3
from bytecode_analyzer import analyze_bytecode_for_ecrecover

def scan_address(rpc_url, address):
    """
    Connects to an Ethereum RPC node, fetches the bytecode for the given address,
    and analyzes it for quantum-vulnerable ecrecover usage.
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"Error: Could not connect to RPC node at {rpc_url}")
        return

    address = Web3.to_checksum_address(address)
    print(f"Fetching bytecode for {address}...")
    
    try:
        bytecode = w3.eth.get_code(address).hex()
    except Exception as e:
        print(f"Error fetching bytecode: {e}")
        return

    if bytecode == "0x" or not bytecode:
        print("Address is an EOA (Externally Owned Account) or has no deployed code.")
        return

    print(f"Bytecode length: {len(bytecode)//2} bytes")
    print("Running Priority 1 Bytecode Scanner...")
    
    result = analyze_bytecode_for_ecrecover(bytecode)
    
    print("\n=========================================")
    print("  QVulnDetect Ethereum-Wide RPC Scanner")
    print("=========================================")
    print(f"Contract Address    : {address}")
    print(f"Instructions Parsed : {result.get('instruction_count', 0)}")
    print(f"ecrecover Detected  : {result.get('detected', False)}")
    
    if result.get('reasons'):
        print("Detection Evidence  :")
        for r in result['reasons']:
            print(f"  -> {r}")
    print("=========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ethereum-Wide Quantum Vulnerability Scanner")
    parser.add_argument("--rpc", type=str, default="https://cloudflare-eth.com", 
                        help="Ethereum RPC URL (default: cloudflare-eth public node)")
    parser.add_argument("--address", type=str, required=True, 
                        help="Smart contract address to scan")
    
    args = parser.parse_args()
    scan_address(args.rpc, args.address)
