import socket

def udp_receiver(host='0.0.0.0', port=12352, buffer_size=4096):
    """
    UDPパケットを受信する関数。

    Parameters:
        host (str): 受信するホストアドレス（デフォルト: '0.0.0.0'）
        port (int): 受信ポート番号（デフォルト: 5005）
        buffer_size (int): 受信バッファサイズ（デフォルト: 1024バイト）

    実行後、指定ポートでUDPパケットを待ち受け、受信ごとに表示します。
    """

    # ソケットを作成（AF_INETはIPv4, SOCK_DGRAMはUDP）
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        print(f"[+] UDPポート {port} で受信待機中...")

        while True:
            data, addr = sock.recvfrom(buffer_size)  # データと送信元アドレスを受信
            print(f"[受信] {addr[0]}:{addr[1]} → {data.decode(errors='ignore')}")

# 実行（ポート番号は適宜変更可能）
if __name__ == "__main__":
    udp_receiver()