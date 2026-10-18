---

<p align="justify"><h1>Projeto95 : Physics-Informed Neural Network (PINN) para Previsão de Demanda e Controle de Estoque no Varejo</h1></p>

<p align="justify">Este documento consolida a implementação e os resultados de uma <b>Physics-Informed Neural Network (PINN)</b> desenvolvida para a previsão conjunta de demanda de canal (<b>Sell-In</b> e <b>Sell-Out</b>) e otimização da dinâmica de inventário no varejo. Ao contrário de modelos puramente estatísticos ou de aprendizado profundo tradicionais, esta abordagem integra diretamente as leis de conservação de massa (equação de balanço de estoque) como restrições físicas no espaço de perda da rede neural, assegurando previsões coerentes com a realidade operacional.</p>

<p align="justify"><h3>1. Fundamentação Teórica e Equação de Balanço</h3></p>

<p align="justify">A modelagem convencional muitas vezes trata o <b>Sell-In</b> (vendas do fabricante para o varejista) e o <b>Sell-Out</b> (vendas do varejista para o consumidor final) de forma isolada, gerando distorções em relação à capacidade física do inventário. A arquitetura PINN incorpora a invariante física de conservação de estoque ao longo do tempo:</p>

<p align="center">
<b>E<sub>t+1</sub> = E<sub>t</sub> + S<sub>in</sub> - S<sub>out</sub></b>
</p>

<p align="justify">Onde <b>E<sub>t</sub></b> representa o estoque inicial no período <b>t</b>, <b>S<sub>in</sub></b> o volume de Sell-In recebido, <b>S<sub>out</sub></b> a saída de Sell-Out para os clientes e <b>E<sub>t+1</sub></b> o nível de estoque resultante. O descumprimento desta igualdade gera uma penalização direta no cálculo dos gradientes durante o processo de <i>backpropagation</i>.</p>

<p align="justify"><h3>2. Estratégia de Modelação e Arquitetura</h3></p>

<p align="justify">Para mitigar instabilidades de otimização e garantir robustez na generalização entre múltiplos produtos e locais, a rede emprega as seguintes diretrizes estruturais:</p>

* <p align="justify"><b>Embeddings Categóricos:</b> Vetores densos de dimensão 32 (<code>emb_dim=32</code>) dedicados a SKUs e Lojas, capturando padrões ocultos de comportamento e elasticidade por localização e categoria.</p>
* <p align="justify"><b>Engenharia de Features Temporais:</b> Inclusão de inércia histórica por meio de <i>lags</i> (<code>sell_out_lag1</code>, <code>sell_in_lag1</code>), médias móveis de 3 períodos (<code>rolling_sellout</code>), além de variáveis macroeconômicas e promocionais.</p>
* <p align="justify"><b>Normalização e Ativação Estável:</b> Utilização de <code>LayerNorm</code> e função de ativação <code>Softplus</code> na camada final, eliminando o problema de gradiente nulo (<i>Dying ReLU</i>) e garantindo predições estritamente positivas.</p>
* <p align="justify"><b>Balanceamento de Perdas no Espaço Padronizado:</b> A restrição física é computada na escala padronizada (Z-score) do inventário, equalizando as ordens de grandeza dos gradientes de dados e de física.</p>

<p align="justify"><h3>3. Função de Custo, Physics Annealing e Detalhamento das Variáveis</h3></p>

<p align="justify">A função de perda total combina o erro empírico dos dados com penalidades físicas e de ruptura de estoque. Para impedir que o rigor físico sufoque o aprendizado estatístico inicial das tendências de mercado, aplicou-se a técnica de <b>Physics Annealing</b>:</p>

<p align="center">
<b>Loss<sub>Total</sub> = Loss<sub>Data</sub> + λ<sub>phy</sub>(t) · Loss<sub>Phy</sub> + λ<sub>ruptura</sub> · Loss<sub>Ruptura</sub></b>
</p>

<p align="justify">Abaixo detalhamos o significado e a função técnica de cada componente da equação de custo:</p>

* <p align="justify"><b>Loss<sub>Data</sub> (Perda Empírica dos Dados):</b> Mede o Erro Quadrático Médio (MSE) entre as previsões do modelo (Sell-In e Sell-Out) e os valores reais observados na base histórica. Garante que o modelo aprenda os padrões de comportamento da demanda comercial.</p>
* <p align="justify"><b>Loss<sub>Phy</sub> (Perda do Balanço Físico):</b> Avalia o grau de desacordo com a equação estrita de conservação de massa do inventário (<b>E<sub>t+1</sub> = E<sub>t</sub> + S<sub>in</sub> - S<sub>out</sub></b>). Penaliza matematicamente qualquer inconsistência no fluxo físico do estoque.</p>
* <p align="justify"><b>λ<sub>phy</sub>(t) (Fator de Ponderação Física com Annealing):</b> Coeficiente dinâmico que decresce ao longo do tempo (épocas <i>t</i> de treinamento). No início do aprendizado, um valor elevado impõe fortemente as leis físicas; conforme a rede estabiliza, o parâmetro decai, permitindo o refinamento empírico da precisão dos dados.</p>
* <p align="justify"><b>Loss<sub>Ruptura</sub> (Penalidade por Ruptura de Estoque):</b> Penalização aplicada caso a demanda estimada de <i>Sell-Out</i> ultrapasse a quantidade de estoque fisicamente disponível (<b>E<sub>t</sub></b>), forçando o modelo a respeitar a capacidade máxima de atendimento operacional.</p>
* <p align="justify"><b>λ<sub>ruptura</sub> (Peso da Ruptura):</b> Hiperparâmetro fixo de calibração que controla o rigor do castigo aplicado a situações de falta iminente de produto frente às demais perdas.</p>

<p align="justify"><h3>4. Resultados Empíricos e Métricas de Desempenho</h3></p>

<p align="justify">O pipeline de treinamento foi executado ao longo de 800 épocas utilizando o otimizador Adam e o escalonador de taxa de aprendizado <code>ReduceLROnPlateau</code>. O modelo alcançou convergência estável com os seguintes resultados obtidos em unidades reais de produtos:</p>

* <p align="justify"><b>RMSE Final (Sell-In):</b> 17.06 unidades reais.</p>
* <p align="justify"><b>RMSE Final (Sell-Out):</b> 16.21 unidades reais.</p>
* <p align="justify"><b>Evolução da Loss de Dados:</b> Redução consistente de 1.5222 para aproximadamente 0.4985 ao término do ciclo.</p>
* <p align="justify"><b>Convergência da Loss Física:</b> Estabilizada em patamares inferiores a ~0.0109, atestando a aderência quase perfeita ao balanço dinâmico de inventário.</p>

<p align="justify"><h3>5. Descrição e Explicação dos Blocos de Código (PINN.ipynb)</h3></p>

<p align="justify"><b>1. Configuração e Device:</b> Detecta o ambiente de execução disponível (GPU via CUDA ou CPU) para alocar os tensores e o modelo eficientemente, além de fixar sementes aleatórias para garantir a reprodutibilidade dos experimentos.</p>

<p align="justify"><b>2. Geração de Dados Sintéticos (Série Temporal e Sazonalidade):</b> Simula um cenário realista de varejo contemplando 20 SKUs e 10 lojas ao longo de 52 semanas. O bloco calcula a demanda base, adiciona um componente senoidal para modelar a sazonalidade anual, insere fatores estocásticos de preço e promoções binárias, e calcula o fluxo de saída (Sell-Out), entrada (Sell-In) e o balanço físico do estoque resultante para o período seguinte (E<sub>t+1</sub>).</p>

<p align="justify"><b>3. Engenharia de Features (Lags, Médias Móveis e Encoding):</b> Prepara as variáveis explicativas (features) para o modelo. Cria <i>lags</i> temporais (<code>sell_out_lag1</code>, <code>sell_in_lag1</code>) e média móvel de 3 períodos (<code>rolling_sellout</code>) para capturar a inércia histórica das vendas. Utiliza <code>LabelEncoder</code> para transformar identificadores textuais de SKUs e Lojas em índices inteiros e aplica <code>StandardScaler</code> para normalizar as variáveis numéricas.</p>

<p align="justify"><b>4. Arquitetura Avançada (PINN.DemandaAdvanced):</b> Define a estrutura da rede neural baseada em PyTorch. Converte os IDs de SKU e Loja em vetores densos de dimensão 32 (<code>emb_dim=32</code>), combina as variáveis numéricas escalonadas com os embeddings e os passa por camadas lineares (256 e 128 neurônios). Utiliza <code>LayerNorm</code> para estabilizar o treinamento e a função de ativação <code>Softplus</code> na camada de saída, assegurando predições estritamente positivas e eliminando o problema de gradientes mortos (<i>Dying ReLU</i>).</p>

<p align="justify"><b>5. Função de Perda Física com Annealing Dinâmico (loss_total):</b> Incorpora as leis físicas de conservação de massa diretamente na otimização da rede. Calcula o MSE entre as previsões e os valores reais, valida a equação de balanço de estoque (E<sub>t+1</sub> = E<sub>t</sub> + S<sub>in</sub> - S<sub>out</sub>) no espaço padronizado (Z-score) e aplica uma penalidade de ruptura de estoque. O peso dinâmico decrescente (<code>lambda_phy</code>) permite balancear o rigor físico com o ajuste empírico dos dados.</p>

<p align="justify"><b>6. Loop de Treinamento Otimizado:</b> Executa o aprendizado dos pesos da rede ao longo de 800 épocas. Utiliza o otimizador Adam em conjunto com o escalonador de taxa de aprendizado <code>ReduceLROnPlateau</code>, reduzindo a taxa de aprendizado caso a perda estagne para garantir convergência suave.</p>

<p align="justify"><b>7. Avaliação de Desempenho (RMSE em Unidades Reais):</b> Mensura a acurácia preditiva do modelo fora da escala normalizada. Realiza a inversão do scaling (<code>scaler_y.inverse_transform</code>) das predições e calcula a Raiz do Erro Quadrático Médio (RMSE) separadamente para Sell-In e Sell-Out em unidades reais de produtos.</p>

<p align="justify"><b>8. Função de Inferência (prever):</b> Disponibiliza uma interface prática para estimar futuras demandas operacionais. Recebe o código de um SKU, de uma loja e um dicionário com as variáveis do período, processa as transformações e retorna os volumes previstos de Sell-In e Sell-Out.</p>

<p align="justify"><h3>Conclusão</h3></p>

<p align="justify">A integração de Redes Neurais Guiadas pela Física (<b>PINN</b>) no planejamento de demanda e controle de estoque demonstra que a inserção de leis físicas fundamentais atua como um regularizador robusto para modelos de aprendizado profundo. Além de reduzir o erro quadrático médio de previsão para patamares altamente competitivos, a metodologia garante a consistência operacional e financeira dos planos de suprimento e reposição no varejo.</p>
