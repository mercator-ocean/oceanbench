import click


@click.command()
def oceanbench_cli():
    click.echo("Ocean Bench command line interface tool")


if __name__ == "__main__":
    oceanbench_cli()
