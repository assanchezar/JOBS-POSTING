from __future__ import annotations

from app.services.email_extract import (
    EmailHit,
    clean_email,
    clean_ocr_text,
    extract_emails,
    merge_email_hits,
    normalize_text,
)


def test_basic_email() -> None:
    hits = extract_emails("Contacto: ventas@acme.com")
    assert [h.email for h in hits] == ["ventas@acme.com"]
    assert hits[0].confidence == 0.85


def test_multiple_emails_in_one_text() -> None:
    hits = extract_emails("ventas@acme.com / soporte@acme.com")
    assert {h.email for h in hits} == {"ventas@acme.com", "soporte@acme.com"}


def test_obfuscated_email() -> None:
    hits = extract_emails("juan.perez [at] empresa [dot] com")
    assert [h.email for h in hits] == ["juan.perez@empresa.com"]


def test_obfuscated_spanish_brackets() -> None:
    hits = extract_emails("juan [ arroba ] empresa [ punto ] com")
    assert [h.email for h in hits] == ["juan@empresa.com"]


def test_accepts_plus_addressing() -> None:
    assert clean_email("ventas+pedidos@acme.com") == "ventas+pedidos@acme.com"


def test_rejects_consecutive_dots_in_local_part() -> None:
    assert extract_emails("a..b@acme.com") == []


def test_rejects_decorative_symbol_as_local_part() -> None:
    assert extract_emails("Instagram |@enreioros.ok") == []
    assert clean_email("|@enreioros.ok") is None


def test_rejects_domain_label_too_long() -> None:
    assert clean_email(f"a@{'a' * 64}.com") is None


def test_rejects_domain_too_long() -> None:
    assert clean_email(f"a@{'a.' * 127}com") is None


def test_fullwidth_characters() -> None:
    hits = extract_emails("ｖｅｎｔａｓ＠ａｃｍｅ．ｃｏｍ")
    assert [h.email for h in hits] == ["ventas@acme.com"]


def test_ocr_copyright_symbol_as_at() -> None:
    hits = extract_emails("ventas©acme.com")
    assert [h.email for h in hits] == ["ventas@acme.com"]


def test_repairs_at_read_as_zero_with_known_domain() -> None:
    hits = extract_emails("recursoshumanosvm190 gmail.com")
    assert [h.email for h in hits] == ["recursoshumanosvm19@gmail.com"]


def test_repairs_at_read_as_zero_with_email_context() -> None:
    hits = extract_emails("Enviar cv a ventas0 empresa.com.ar")
    assert [h.email for h in hits] == ["ventas@empresa.com.ar"]


def test_does_not_repair_zero_without_email_context() -> None:
    assert extract_emails("modelo0 empresa.com") == []


def test_does_not_repair_numeric_only_local_part() -> None:
    assert extract_emails("cv 3510 gmail.com") == []


def test_clean_ocr_text_removes_decorative_symbols() -> None:
    text = "MUNDOPTICO\n@\n@@@\n---\n\ncontacto@acme.com\nHola"
    assert clean_ocr_text(text) == "MUNDOPTICO\n\ncontacto@acme.com\nHola"


def test_clean_ocr_text_collapses_repeated_lines() -> None:
    assert clean_ocr_text("Aviso\nAviso\nAviso\nTexto") == "Aviso\nTexto"


def test_split_around_at_symbol() -> None:
    hits = extract_emails("juan.perez@ empresa.com")
    assert [h.email for h in hits] == ["juan.perez@empresa.com"]


def test_split_around_dot() -> None:
    hits = extract_emails("ana@empresa . com")
    assert [h.email for h in hits] == ["ana@empresa.com"]


def test_trailing_punctuation() -> None:
    hits = extract_emails("Escribe a info@acme.org, gracias.")
    assert [h.email for h in hits] == ["info@acme.org"]


def test_hyphenated_domain() -> None:
    hits = extract_emails("facturas@mi-empresa.org")
    assert [h.email for h in hits] == ["facturas@mi-empresa.org"]


def test_dedupe_is_case_insensitive() -> None:
    hits = extract_emails("Ventas@Acme.com\nventas@acme.com")
    assert len(hits) == 1
    assert hits[0].occurrences == 2


def test_custom_confidence() -> None:
    hits = extract_emails("ventas@acme.com", confidence=0.7)
    assert hits[0].confidence == 0.7


def test_rejects_file_extension_as_tld() -> None:
    assert clean_email("imagen@logo.png") is None


def test_repairs_common_tld_ocr_confusions() -> None:
    assert clean_email("ventas@acme.c0m") == "ventas@acme.com"
    assert clean_email("soporte@acme.c0rn") == "soporte@acme.com"
    assert clean_email("info@acme.n3t") == "info@acme.net"
    assert clean_email("admin@acme.0rg") == "admin@acme.org"
    assert clean_email("hola@acme.1nfo") == "hola@acme.info"


def test_does_not_modify_valid_tlds() -> None:
    assert clean_email("a@acme.cm") == "a@acme.cm"
    assert clean_email("a@acme.com") == "a@acme.com"


def test_repairs_tld_before_deduplicating() -> None:
    hits = extract_emails("ventas@acme.c0m\nventas@acme.com")
    assert len(hits) == 1
    assert hits[0].email == "ventas@acme.com"


def test_rejects_domain_without_dot() -> None:
    assert clean_email("usuario@localhost") is None


def test_rejects_numeric_tld() -> None:
    assert clean_email("usuario@192.168.1.1") is None


def test_rejects_underscore_in_domain() -> None:
    assert clean_email("usuario@mi_empresa.com") is None


def test_normalize_removes_zero_width() -> None:
    assert normalize_text("ven\u200btas@acme.com") == "ventas@acme.com"


def test_no_false_positive_without_email() -> None:
    assert extract_emails("Reunion a las 10:30 en la sala 3") == []


def test_merge_prefers_higher_confidence_and_occurrences() -> None:
    model = [EmailHit("Ventas@Acme.com", 0.95, 1)]
    text = [EmailHit("ventas@acme.com", 0.85, 3)]
    merged = merge_email_hits(model, text)
    assert len(merged) == 1
    assert merged[0].email == "Ventas@Acme.com"
    assert merged[0].confidence == 0.95
    assert merged[0].occurrences == 3


def test_merge_sorts_by_confidence_then_email() -> None:
    merged = merge_email_hits(
        [EmailHit("b@acme.com", 0.5, 1)],
        [EmailHit("a@acme.com", 0.9, 1), EmailHit("c@acme.com", 0.5, 1)],
    )
    assert [hit.email for hit in merged] == ["a@acme.com", "b@acme.com", "c@acme.com"]
