import click

from oakeye.cli.trinocular import trinocular

@click.group('oakeye')
def cli():
    pass

cli.add_command(trinocular)

if __name__ == '__main__':
    cli()