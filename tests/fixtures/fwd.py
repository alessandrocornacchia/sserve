#!/usr/bin/env python3
"""fwd.py LPORT HOST PORT: TCP forwarder standing in for `ssh -N -L 127.0.0.1:LPORT:HOST:PORT`.
Killing it closes the tunnel and every connection through it, like ssh."""
import asyncio
import sys


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection(sys.argv[2], int(sys.argv[3]))
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", int(sys.argv[1]))
    async with srv:
        await srv.serve_forever()


asyncio.run(main())
