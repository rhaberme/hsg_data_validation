import streamlit as st

st.set_page_config(
    page_title="HSG Sim Datatool",
    page_icon="HSG",
    layout="centered"
)

hide_menu = """
<style>
#MainMenu {visibility:hidden;}
footer{visibility:hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
st.markdown("""
<style>
.small-font-green {
    font-size:12px;
    color: green;
}
.small-font-red {
    font-size:12px;
    color: red;
}
.normal-font-red {
    font-size:16px;
    color: red;
}
.normal-font-green {
    font-size:16px;
    color: green;
}
</style>
""", unsafe_allow_html=True)

# Main area content
st.title("Nutzungsbedingungen")
st.write("""Nutzungsbedingungen für HSGSim Data Tool Stand: 17. Juni 2025 § 1 Geltungsbereich und Anbieter (1) Diese Nutzungsbedingungen regeln die Nutzung des auf der Webseite https://hsgdatatool.streamlit.app/ unentgeltlich zur Verfügung gestellten Online-Tools (nachfolgend „Dienst“). (2) Anbieter des Dienstes ist: Hochschulgruppe Simulation (nachfolgend „Anbieter“). (3) Mit dem Hochladen von Daten und der Nutzung des Dienstes erklärt sich der Nutzer mit diesen Nutzungsbedingungen einverstanden. Abweichende Bedingungen des Nutzers werden nicht anerkannt, es sei denn, der Anbieter stimmt ihrer Geltung ausdrücklich schriftlich zu. § 2 Leistungsgegenstand (1) Der Anbieter stellt dem Nutzer ein Online-Tool zur Verfügung, mit dem der Nutzer anonyme Messdaten (nachfolgend „Daten“) hochladen und automatisiert verarbeiten lassen kann. Die Art der Verarbeitung und die möglichen Ergebnisse (z.B. Analyse, Visualisierung, Konvertierung) werden auf der Webseite näher beschrieben. (2) Der Dienst wird ausschließlich unentgeltlich zur Verfügung gestellt. Es besteht kein Anspruch auf Registrierung oder Nutzung. (3) Der Anbieter ist berechtigt, den Dienst jederzeit und ohne Angabe von Gründen zu ändern, zu beschränken oder vollständig einzustellen. § 3 Pflichten und Verantwortlichkeiten des Nutzers (1) Der Nutzer ist für die von ihm hochgeladenen Daten allein verantwortlich. (2) Der Nutzer sichert zu, dass die von ihm hochgeladenen Daten keine personenbezogenen Daten im Sinne der Datenschutz-Grundverordnung (DSGVO) enthalten und auch keine Rückschlüsse auf identifizierbare Personen zulassen. Das Hochladen von personenbezogenen Daten ist strengstens untersagt. (3) Der Nutzer sichert weiterhin zu, dass die von ihm bereitgestellten Daten frei von Rechten Dritter (insbesondere Urheber- und Leistungsschutzrechten) sind, die einer Verarbeitung im Rahmen dieses Dienstes entgegenstehen. (4) Es ist dem Nutzer untersagt, den Dienst missbräuchlich zu nutzen. Dies umfasst insbesondere das Hochladen von Daten, die Viren, Trojaner oder andere schädliche Software enthalten, sowie Versuche, die Funktionsweise des Dienstes zu stören oder zu umgehen. (5) Der Nutzer ist dafür verantwortlich, Sicherungskopien der von ihm hochgeladenen Daten zu erstellen. Der Anbieter übernimmt keine Verantwortung für einen etwaigen Datenverlust.

§ 4 Einräumung von Nutzungsrechten (1) Der Nutzer räumt dem Anbieter mit dem Hochladen der Daten ein nicht-ausschließliches, weltweites und unentgeltliches Recht ein, die hochgeladenen anonymen Daten für die Dauer der Nutzung technisch zu vervielfältigen, zu bearbeiten und zu verarbeiten, soweit dies zur Erbringung des Dienstes erforderlich ist. (2) Der Anbieter ist berechtigt, die hochgeladenen sowie die durch den Dienst generierten anonymen Daten in vollständig anonymisierter Form zu speichern und für eigene Zwecke, wie beispielsweise zur Verbesserung des Dienstes, für statistische Analysen oder für wissenschaftliche Auswertungen, zu nutzen. Ein Rückschluss auf den ursprünglichen Nutzer ist hierbei ausgeschlossen. § 5 Verfügbarkeit Der Anbieter bemüht sich, den Dienst möglichst unterbrechungsfrei zur Verfügung zu stellen. Der Nutzer hat jedoch keinen Anspruch auf eine ständige und ununterbrochene Verfügbarkeit. Der Anbieter übernimmt keine Gewährleistung für die Erreichbarkeit des Dienstes und schließt jegliche Haftung für Ausfallzeiten aus. § 6 Gewährleistung und Haftung (1) Der Dienst wird „wie besehen“ und ohne jegliche Mängelgewährleistung zur Verfügung gestellt. Der Anbieter übernimmt keine Gewähr dafür, dass die durch den Dienst generierten Ergebnisse korrekt, vollständig oder für den vom Nutzer verfolgten Zweck geeignet sind. Die Nutzung der Ergebnisse erfolgt auf eigenes Risiko des Nutzers. (2) Da der Dienst unentgeltlich erbracht wird, ist die Haftung des Anbieters stark eingeschränkt. Der Anbieter haftet unbeschränkt nur für Vorsatz und grobe Fahrlässigkeit. (3) Für leichte Fahrlässigkeit haftet der Anbieter nur bei Verletzung einer wesentlichen Vertragspflicht (Kardinalpflicht), deren Erfüllung die ordnungsgemäße Durchführung des Vertrags überhaupt erst ermöglicht und auf deren Einhaltung der Nutzer regelmäßig vertrauen darf. In diesem Fall ist die Haftung auf den vertragstypischen, vorhersehbaren Schaden begrenzt. (4) Die vorstehenden Haftungsbeschränkungen gelten nicht bei Verletzung von Leben, Körper oder Gesundheit, bei arglistig verschwiegenen Mängeln oder bei einer Haftung nach dem Produkthaftungsgesetz. (5) Soweit die Haftung des Anbieters ausgeschlossen oder beschränkt ist, gilt dies auch für die persönliche Haftung von Arbeitnehmern, Vertretern und Erfüllungsgehilfen des Anbieters. § 7 Schlussbestimmungen (1) Es gilt das Recht der Bundesrepublik Deutschland unter Ausschluss des UN-Kaufrechts. (2) Sollten einzelne Bestimmungen dieser Nutzungsbedingungen ganz oder teilweise unwirksam sein oder werden, so wird hierdurch die Gültigkeit der übrigen Bestimmungen nicht berührt. An die Stelle der unwirksamen Bestimmung tritt die gesetzliche Regelung. (3) Der Anbieter behält sich das Recht vor, diese Nutzungsbedingungen zu ändern. Die Nutzer werden über Änderungen nicht individuell informiert. Es obliegt dem Nutzer, die Nutzungsbedingungen vor jeder Nutzung auf deren Aktualität zu prüfen. Die fortgesetzte Nutzung des Dienstes nach einer Änderung gilt als Zustimmung zu den neuen Bedingungen.""")