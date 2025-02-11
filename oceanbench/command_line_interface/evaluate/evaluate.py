import click

from oceanbench.command_line_interface.evaluate.rmse import pointwise_evaluation


@click.group()
def evaluate():
    click.echo("Using evaluate command...")


evaluate.add_command(pointwise_evaluation)
