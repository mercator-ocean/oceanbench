import click

from oceanbench.command_line_interface.process.calc_mld import calc_mld


@click.group()
def process():
    click.echo("Using process command...")


process.add_command(calc_mld)
