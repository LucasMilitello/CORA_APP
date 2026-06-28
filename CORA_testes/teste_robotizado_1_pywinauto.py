"""Teste 1: usa uma interface CORA ja aberta e salva todas as imagens.

Todas as acoes humanas mantem o intervalo padrao de 1 segundo. Execute
diretamente no VS Code com a tela inicial do CORA aberta.
"""

from teste_robotizado_comum import (
    RobotizedTestContext,
    SAVE_SELECTION_DELAY_SECONDS,
    mark_all_images_for_saving,
    run_robotized_test,
)



def execute_test_1(context: RobotizedTestContext) -> None:
    """Percorre os grupos e marca cada imagem em 'Incluir no salvamento'."""
    marked = mark_all_images_for_saving(
        context.main_window,
        context.robot,
        total_images=len(context.source_images),
    )
    expected = len(context.source_images)
    if marked != expected:
        raise RuntimeError(
            f"Nem todas as imagens foram marcadas para salvar: {marked}/{expected}."
        )
    print(f"Imagens marcadas: {marked}", flush=True)


def main() -> int:
    return run_robotized_test(
        test_number=1,
        interval_description=(
            f"intervalo entre cliques de selecao: {SAVE_SELECTION_DELAY_SECONDS:.1f}s"
        ),
        execute_specific_flow=execute_test_1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
