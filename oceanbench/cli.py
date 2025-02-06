import click


@click.command()
def run():
    click.echo("Ocean Bench command line interface tool")


if __name__ == "__main__":
    run()
