"""Teste 2: usa uma interface CORA ja aberta e edita todas as imagens.

Cada imagem permanece 15 segundos no editor antes da confirmacao. As demais
acoes mantem o intervalo padrao de 1 segundo. Execute diretamente no VS Code
com a tela inicial do CORA aberta.
"""

from teste_robotizado_comum import (
    EDIT_IMAGE_INTERVAL_SECONDS,
    RobotizedTestContext,
    edit_every_image,
    run_robotized_test,
    visit_all_groups,
    wait_and_answer_dialog,
    wait_for_robot_edit_completion,
)


def execute_test_2(context: RobotizedTestContext) -> None:
    """Visita os grupos e confirma cada mascara no editor continuo."""
    visited = visit_all_groups(
        context.main_window,
        context.robot,
        maximum=max(10, len(context.source_images) + 1),
    )
    print(f"Grupos visitados: {visited}", flush=True)

    context.main_window.set_focus()
    context.robot.keys("+{F7}", "Abrir a edicao de todas as mascaras")
    wait_and_answer_dialog(context.application, context.robot, prefer_yes=True, timeout=30)
    edited = edit_every_image(
        context.application,
        context.robot,
        maximum_images=len(context.source_images) + 5,
    )
    expected = len(context.source_images)
    if edited != expected:
        raise RuntimeError(
            f"Nem todas as imagens foram confirmadas no editor: {edited}/{expected}."
        )
    print(f"Mascaras confirmadas: {edited}", flush=True)

    wait_for_robot_edit_completion(
        context.application,
        context.main_window,
        context.robot,
        expected_images=expected,
        timeout=context.processing_timeout,
    )


def main() -> int:
    return run_robotized_test(
        test_number=2,
        interval_description=(
            f"tempo simulado de edicao por imagem: {EDIT_IMAGE_INTERVAL_SECONDS:.1f}s"
        ),
        execute_specific_flow=execute_test_2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
