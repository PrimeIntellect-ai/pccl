from pccl import MasterNode
import argparse
import logging
import signal

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='PCCL Master Node')
    parser.add_argument(
        '--listen-address',
        type=str,
        default='0.0.0.0:48148',
        help='Address for the master node to listen on (format: host:port)'
    )

    args = parser.parse_args()

    logging.info(f"Starting master node on {args.listen_address}")
    master: MasterNode = MasterNode(listen_address=args.listen_address)

    def _graceful_shutdown(signum, frame):
        logging.info("Received Ctrl-C / SIGINT. Shutting down master...")
        master.interrupt()

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        master.run()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught. Shutting down master...")
        master.interrupt()


if __name__ == '__main__':
    main()
