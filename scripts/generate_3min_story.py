"""Generate full 3-minute (180.0s) cinematic master clip for 'The Clockwork Alchemist'.

Constructs the complete 30-shot dramatic narrative with Aria Vance and Archivist Malcolm,
anchors each shot to canonical character turnaround sheets, renders keyframes/videos,
and assembles the final 3-minute movie with audio.
"""

from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from pipeline.assemble import build_concat_list_file, concatenate_clips
from pipeline.character_sheet import ensure_character_sheets
from pipeline.generate import process_single_shot
from pipeline.models import CharacterProfile, DialogueLine, ShotJob
from pipeline.state_manager import (
    get_character_dict,
    init_database,
    list_shots,
    save_characters_batch,
    save_shots_batch,
)

console = Console()

DB_PATH = "output/shots.db"
OUTPUT_DIR = "output"


def build_3min_shot_list() -> list[ShotJob]:
    """Builds 30 sequential cinematic shots of 6.0s each (180.0s total)."""
    shot_specs = [
        # Act I: The Midnight Observatory & The Warning (Shots 1-10)
        (
            1, "wide", None, None,
            "Gothic observatory tower overlooking the rain-swept spires of Oakhaven under midnight lightning.",
            "atmospheric",
        ),
        (
            2, "medium", None, None,
            "Interior observatory cluttered with brass astrolabes, glowing mercury vials, and antique stellar charts.",
            "mysterious",
        ),
        (
            3, "close_up", "aria", "The harmonic frequency is nearly locked into resonance.",
            "Close-up of Aria Vance tightening calipers around the pulsating cobalt starlight core, sparks reflecting in her goggles.",
            "focused",
        ),
        (
            4, "medium", None, None,
            "Shadow beside the towering pendulum clock lengthens as Archivist Malcolm steps into the amber lantern light.",
            "suspenseful",
        ),
        (
            5, "close_up", "malcolm", "You are tempting currents you cannot siphon back into the earth, Aria.",
            "Archivist Malcolm frowning in solemn contemplation, clutching his antique pocket watch with a weathered glove.",
            "grave",
        ),
        (
            6, "medium", "aria", "Fear is an anchor for minds without vision, Malcolm.",
            "Aria turning slightly toward Malcolm with steady hands on the vernier mechanism, undaunted.",
            "defiant",
        ),
        (
            7, "close_up", "aria", "The astrolabe was designed to measure the resonant frequency of the heavens.",
            "Tight close-up on Aria's resolute dark eyes glowing with cobalt reflection.",
            "visionary",
        ),
        (
            8, "medium", "malcolm", "And if the heavens answer? You risk repeating the disaster of the Silent Dawn.",
            "Malcolm stepping closer to the mahogany workbench, midnight rain drumming hard against the arched glass.",
            "urgent",
        ),
        (
            9, "close_up", "malcolm", "The Grand Guild shattered every pane of glass across the continent.",
            "Malcolm's weathered features bathed in deep blue radiance as he recalls the ancient catastrophe.",
            "solemn",
        ),
        (
            10, "close_up", "aria", "The Guild lacked the stabilized prism. They forced the current. I am inviting it.",
            "Aria smiling with calm scientific confidence, hand hovering over the master brass bezel.",
            "confident",
        ),

        # Act II: The Mechanical Ignition & Gravitational Inversion (Shots 11-20)
        (
            11, "close_up", None, None,
            "Aria's gloved fingers firmly rotating the engraved brass bezel ninety degrees clockwise.",
            "climactic",
        ),
        (
            12, "wide", None, None,
            "Interlocking gear trains engage; concentric brass astrolabe rings begin spinning in rapid counter-rotation.",
            "dynamic",
        ),
        (
            13, "medium", None, None,
            "A sharp crystalline harmonic chime resonates through the observatory chamber, vibrating glassware on wooden shelves.",
            "harmonious",
        ),
        (
            14, "close_up", None, None,
            "Liquid mercury droplets in glass capillary tubes defy gravity, rising upward into shimmering suspension.",
            "wondrous",
        ),
        (
            15, "medium", "malcolm", "The chronometer... its gears are spinning in reverse.",
            "Malcolm gazing in disbelief at his antique pocket watch spinning backward on its silver chain.",
            "shocked",
        ),
        (
            16, "close_up", None, None,
            "Macro view of the ornate watch face as the filigree hands whirl counter-clockwise at blinding speed.",
            "surreal",
        ),
        (
            17, "medium", "aria", "Gravitational flux inversion. The harmonic field has stabilized!",
            "Aria watching the levitating dust motes, her expression radiant with triumphant discovery.",
            "exhilarated",
        ),
        (
            18, "wide", None, None,
            "The central sapphire prism emits a vertical column of pure cobalt energy directly toward the glass dome apex.",
            "spectacular",
        ),
        (
            19, "medium", None, None,
            "The leaded cathedral glass above begins glowing with intricate geometric fractal light patterns.",
            "ethereal",
        ),
        (
            20, "close_up", "malcolm", "The ancient codices were true. The celestial conduit is real.",
            "Malcolm's skepticism dissolving into profound spiritual awe as he looks up into the glowing light.",
            "reverent",
        ),

        # Act III: The Celestial Descent & The New Age (Shots 21-30)
        (
            21, "wide", None, None,
            "Exterior shot of the gothic observatory; storm clouds part into a majestic spiraling celestial vortex.",
            "epic",
        ),
        (
            22, "medium", None, None,
            "Raindrops halt mid-air outside the curved glass dome, hanging suspended like thousands of diamond stars.",
            "magical",
        ),
        (
            23, "close_up", "aria", "Look up, Malcolm. The stars are answering.",
            "Aria gazing through the open aperture of the dome, eyes shimmering with tearful wonder.",
            "transcendent",
        ),
        (
            24, "wide", None, None,
            "A radiant beam of starlight descends from the heavens, piercing the storm clouds toward the observatory.",
            "monumental",
        ),
        (
            25, "medium", None, None,
            "The celestial beam meets the astrolabe prism, scattering prismatic rainbows across the brass mechanisms.",
            "brilliant",
        ),
        (
            26, "medium", "malcolm", "Then let the new age be written in light, Vance.",
            "Malcolm placing a reverent hand on the console bench, nodding to Aria with deep newfound respect.",
            "affirming",
        ),
        (
            27, "close_up", "aria", "This is only the first harmonic. Tomorrow, we chart the constellations.",
            "Aria smiling warmly, adjusting her goggles as the harmonic beacon shines steadfast and true.",
            "triumphant",
        ),
        (
            28, "wide", None, None,
            "The entire observatory interior glows in golden and cobalt harmony as the astrolabe sings into the night.",
            "harmonious",
        ),
        (
            29, "wide", None, None,
            "Exterior aerial camera pulling back across the rooftops of Oakhaven, bathed in the blue beacon glow.",
            "sweeping",
        ),
        (
            30, "wide", None, None,
            "Grand cinematic panoramic conclusion as dawn breaks over the spire city horizon under calm clear skies.",
            "majestic",
        ),
    ]

    shots = []
    for seq, shot_type, speaker, dialogue_text, prompt, mood in shot_specs:
        dialogue = (
            (DialogueLine(character=speaker, text=dialogue_text),)
            if speaker and dialogue_text
            else ()
        )
        shot = ShotJob(
            shot_id=f"ch01_sc01_sh{seq:03d}",
            sequence_order=seq,
            chapter_num=1,
            scene_num=1,
            shot_type=shot_type,
            characters=(speaker,) if speaker else ("aria", "malcolm"),
            speaker=speaker,
            dialogue=dialogue,
            lip_sync_required=bool(speaker),
            visual_prompt=prompt,
            mood=mood,
            target_duration_sec=6.0,
            status="queued",
            attempts=0,
            seed=100 + seq,
        )
        shots.append(shot)

    return shots


def main():
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan] 🎬 PRODUCING 3-MINUTE MASTER FILM: 'THE CLOCKWORK ALCHEMIST' [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")

    init_database(DB_PATH)

    # 1. Register canonical character profiles
    aria = CharacterProfile(
        char_id="aria",
        name="Aria Vance",
        gender="female",
        age="27",
        appearance_description=(
            "Young alchemical clockwork engineer, silver-streaked dark hair tied back with bronze clasp, "
            "brass-rimmed magnifying goggles resting on forehead, fitted dark leather workshop vest over linen shirt, "
            "smudged hands, luminous dark eyes full of relentless intellect."
        ),
        personality_tone="visionary and defiant",
        voice_timbre="clear resonant youthful voice with a hint of steel",
    )
    malcolm = CharacterProfile(
        char_id="malcolm",
        name="Archivist Malcolm",
        gender="male",
        age="58",
        appearance_description=(
            "Weathered senior stellar cartographer, silver-threaded beard, heavy obsidian velvet scholar coat "
            "with tarnished silver Guild insignia, antique brass pocket watch on silver chain, intense piercing gaze."
        ),
        personality_tone="solemn and protective",
        voice_timbre="grave resonant baritone with slow rhythmic cadence",
    )

    save_characters_batch(DB_PATH, [aria, malcolm])
    console.print("[cyan]Ensuring canonical character sheets and PuLID anchors...[/cyan]")
    char_dict = ensure_character_sheets(DB_PATH, output_dir=OUTPUT_DIR, mock=True)

    for cid, c in char_dict.items():
        if cid in ("aria", "malcolm"):
            console.print(f"  ✓ Character '{c.name}' anchored to: [green]{c.reference_image_paths[0]}[/green]")

    # 2. Build and populate 30 shots (30 x 6.0s = 180.0s = 3 minutes)
    shots = build_3min_shot_list()
    save_shots_batch(DB_PATH, shots)
    total_dur = sum(s.target_duration_sec for s in shots)
    console.print(f"[green]✓ Generated {len(shots)} shots totaling exactly {total_dur:.1f}s ({total_dur/60:.1f} minutes).[/green]")

    # 3. Render all shots sequentially
    console.print(f"[cyan]Rendering {len(shots)} sequential shots with facial consistency anchors...[/cyan]")
    for idx, shot in enumerate(shots, 1):
        processed = process_single_shot(
            shot=shot,
            db_path=DB_PATH,
            visual_profiles=char_dict,
            voice_profiles=char_dict,
            output_dir=OUTPUT_DIR,
            mock_rendering=True,
        )
        console.print(f"  [{idx:02d}/30] Shot {processed.shot_id} -> [green]{processed.status}[/green] ({processed.target_duration_sec}s)")

    # 4. Assemble into master 3-minute video
    console.print("[cyan]Assembling final 3-minute movie timeline using FFmpeg...[/cyan]")
    all_shots = list_shots(DB_PATH, status="passed")
    video_files = [s.synced_path for s in all_shots if s.synced_path and Path(s.synced_path).exists()]

    out_final = Path(OUTPUT_DIR) / "final"
    out_final.mkdir(parents=True, exist_ok=True)
    concat_txt = out_final / "the_clockwork_alchemist_concat.txt"
    final_video = out_final / "the_clockwork_alchemist_3min.mp4"

    build_concat_list_file(video_files, str(concat_txt))
    concatenate_clips(str(concat_txt), str(final_video))

    # Also symlink or copy to default assembled path for the web player
    default_player_vid = out_final / "test_scene_assembled.mp4"
    import shutil
    shutil.copyfile(final_video, default_player_vid)

    console.print(f"[bold green]✓ SUCCESS: Master 3-minute film assembled at: {final_video.resolve()}[/bold green]")


if __name__ == "__main__":
    main()
