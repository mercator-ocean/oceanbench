import click

from oceanbench.command_line_interface.evaluate.evaluate import evaluate
from oceanbench.command_line_interface.plot.plot import plot
from oceanbench.command_line_interface.process.process import process


@click.group()
def run():
    click.echo("Ocean Bench command line interface tool")


run.add_command(evaluate)
run.add_command(plot)
run.add_command(process)

if __name__ == "__main__":
    run()
