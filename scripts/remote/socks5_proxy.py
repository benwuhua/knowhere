#!/usr/bin/env python3

import argparse
import ipaddress
import os
import select
import socket
import sys


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("unexpected EOF from proxy")
        data += chunk
    return data


def socks5_connect(
    proxy_host: str,
    proxy_port: int,
    username: str,
    password: str,
    target_host: str,
    target_port: int,
) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    sock.settimeout(15)

    # Greeting: SOCKS5 + username/password auth.
    sock.sendall(b"\x05\x01\x02")
    ver, method = recv_exact(sock, 2)
    if ver != 0x05 or method != 0x02:
        raise RuntimeError("proxy does not accept username/password auth")

    u = username.encode("utf-8")
    p = password.encode("utf-8")
    if len(u) > 255 or len(p) > 255:
        raise RuntimeError("username/password too long for SOCKS5")
    sock.sendall(bytes([0x01, len(u)]) + u + bytes([len(p)]) + p)
    auth_ver, status = recv_exact(sock, 2)
    if auth_ver != 0x01 or status != 0x00:
        raise RuntimeError("proxy authentication failed")

    # CONNECT request
    try:
        ip = ipaddress.ip_address(target_host)
        if ip.version == 4:
            atyp = 0x01
            addr = ip.packed
        else:
            atyp = 0x04
            addr = ip.packed
    except ValueError:
        host_bytes = target_host.encode("idna")
        if len(host_bytes) > 255:
            raise RuntimeError("target hostname too long")
        atyp = 0x03
        addr = bytes([len(host_bytes)]) + host_bytes

    port = target_port.to_bytes(2, "big")
    sock.sendall(bytes([0x05, 0x01, 0x00, atyp]) + addr + port)

    head = recv_exact(sock, 4)
    ver, rep, _rsv, atyp = head
    if ver != 0x05 or rep != 0x00:
        raise RuntimeError(f"proxy connect failed with code {rep}")

    if atyp == 0x01:
        recv_exact(sock, 4)
    elif atyp == 0x04:
        recv_exact(sock, 16)
    elif atyp == 0x03:
        ln = recv_exact(sock, 1)[0]
        recv_exact(sock, ln)
    else:
        raise RuntimeError("proxy returned invalid address type")
    recv_exact(sock, 2)

    sock.settimeout(None)
    return sock


def relay(sock: socket.socket) -> int:
    sock_fd = sock.fileno()
    in_fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    stdin_open = True

    while True:
        read_fds = [sock_fd]
        if stdin_open:
            read_fds.append(in_fd)
        ready, _, _ = select.select(read_fds, [], [])

        if in_fd in ready:
            data = os.read(in_fd, 65536)
            if data:
                sock.sendall(data)
            else:
                stdin_open = False
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        if sock_fd in ready:
            data = sock.recv(65536)
            if data:
                os.write(out_fd, data)
            else:
                return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-host", required=True)
    parser.add_argument("--proxy-port", required=True, type=int)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("target_host")
    parser.add_argument("target_port", type=int)
    args = parser.parse_args()

    try:
        sock = socks5_connect(
            args.proxy_host,
            args.proxy_port,
            args.username,
            args.password,
            args.target_host,
            args.target_port,
        )
        with sock:
            return relay(sock)
    except Exception as exc:
        print(f"socks5 proxy error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
