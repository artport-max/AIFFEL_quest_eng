import json
import os

# ── Glazy 원본 JSON에서 UMF 산화물 키 목록 ──────────────
OXIDE_KEYS = [
    "SiO2", "Al2O3", "B2O3", "Li2O", "K2O", "Na2O",
    "KNaO", "MgO", "CaO", "SrO", "BaO", "ZnO",
    "TiO2", "ZrO2", "Fe2O3", "CoO", "CuO", "Cr2O3",
    "MnO", "MnO2", "NiO", "P2O5", "LOI"
]

def extract_umf(item: dict) -> dict:
    """
    Glazy 원본 item에서 UMF 산화물 수치 추출.
    가능한 위치를 순서대로 탐색:
      1) item['analysis']['umfAnalysis']
      2) item['analysis']['percentageAnalysis']
      3) item 최상위에 직접 있는 산화물 키
    """
    umf = {}

    # 1순위 — analysis.umfAnalysis
    analysis = item.get("analysis", {})
    if analysis:
        umf_data = analysis.get("umfAnalysis", {})
        if umf_data:
            for key in OXIDE_KEYS:
                val = umf_data.get(key)
                if val is not None and float(val) > 0:
                    umf[key] = round(float(val), 4)

    # 2순위 — analysis.percentageAnalysis (umf 없을 때)
    if not umf and analysis:
        pct_data = analysis.get("percentageAnalysis", {})
        if pct_data:
            for key in OXIDE_KEYS:
                val = pct_data.get(key)
                if val is not None and float(val) > 0:
                    umf[key] = round(float(val), 4)

    # 3순위 — item 최상위 직접 탐색
    if not umf:
        for key in OXIDE_KEYS:
            val = item.get(key)
            if val is not None:
                try:
                    fval = float(val)
                    if fval > 0:
                        umf[key] = round(fval, 4)
                except (ValueError, TypeError):
                    pass

    return umf


def extract_firing_info(item: dict) -> dict:
    """소성 온도 및 분위기 추출"""
    firing = {}

    # 소성 온도 범위
    cone_low  = item.get("fromOrtonCone") or item.get("cone") or ""
    cone_high = item.get("toOrtonCone", "")
    if cone_low:
        firing["cone"] = f"Cone {cone_low}" + (f"~{cone_high}" if cone_high else "")

    # 소성 분위기
    atm = item.get("atmosphereName", "") or item.get("atmosphere", "")
    if atm:
        firing["atmosphere"] = atm

    # 표면 타입
    surface = item.get("surfaceTypeName", "") or item.get("surface", "")
    if surface:
        firing["surface"] = surface

    # 투명도
    transparency = item.get("transparencyName", "") or item.get("transparency", "")
    if transparency:
        firing["transparency"] = transparency

    return firing


def run_preprocessing():
    input_dir   = 'data/raw_glaze'
    output_dir  = 'data/sampled_glaze'
    output_path = os.path.join(output_dir, 'combined_glaze.json')

    os.makedirs(output_dir, exist_ok=True)

    all_refined_data = []
    umf_found    = 0
    umf_missing  = 0

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"총 {len(file_list)}개 파일 처리 시작...")

    for file_name in file_list:
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                item = content.get('data', {})

                if not item:
                    continue

                # 기본 정보
                glaze_name  = item.get('name', '이름 없는 유약')
                description = item.get('description', '설명 없음')
                state       = item.get('materialStateName', '상태 미정')

                # UMF 추출 (신규)
                umf = extract_umf(item)

                # 소성 정보 추출 (신규)
                firing = extract_firing_info(item)

                # 통계
                if umf:
                    umf_found += 1
                else:
                    umf_missing += 1

                # AI 입력 텍스트 생성
                refined_text = (
                    f"유약 이름은 {glaze_name}입니다. "
                    f"현재 {state} 단계이며, "
                    f"상세 설명은 다음과 같습니다: {description}"
                )

                all_refined_data.append({
                    "id":                   item.get('id'),
                    "name":                 glaze_name,
                    "original_description": description,
                    "input_for_ai":         refined_text,
                    "umf":                  umf,       # ← 신규 추가
                    "firing":               firing,    # ← 신규 추가
                    "state":                state,
                })

            except Exception as e:
                print(f"  ⚠️  {file_name} 처리 오류: {e}")

    # 결과 저장
    if all_refined_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_refined_data, f, ensure_ascii=False, indent=4)

        print(f"\n✅ 전처리 완료!")
        print(f"   총 처리:       {len(all_refined_data)}개")
        print(f"   UMF 추출 성공: {umf_found}개")
        print(f"   UMF 없음:      {umf_missing}개")
        print(f"   저장 위치:     {output_path}")

        # 샘플 확인 출력
        sample = all_refined_data[0]
        print(f"\n── 샘플 확인 ({sample['name']}) ──")
        print(f"   UMF: {sample['umf']}")
        print(f"   소성: {sample['firing']}")
    else:
        print("❌ 처리할 데이터가 없습니다.")


if __name__ == "__main__":
    run_preprocessing()
